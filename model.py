from transformers import AutoConfig, Wav2Vec2Processor

from transformers.models.wav2vec2.feature_extraction_wav2vec2 import (
    Wav2Vec2FeatureExtractor,
)
from transformers.file_utils import ModelOutput
import torchaudio

from audio_process import Filter, AcousticFeatureExtractor

from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
)
import numpy as np
from transformers import EvalPrediction, TrainingArguments, Trainer, is_apex_available

if is_apex_available():
    from apex import amp

from packaging import version

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

from torch.nn import CrossEntropyLoss

# TODO Eliminar warnings


def speech_file_to_array_fn(path):
    """Carga un audio y lo resamplea al sample rate del modelo"""
    sr = 16000
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, sr)

    speech = resampler(speech_array).squeeze().numpy()

    return speech


def extract_acoustic_features(y, feature_extractor, filter):
    """Extrae los features acústicos del audio"""
    sr = 16000
    y = filter.process(y, sr)
    features = feature_extractor.extract_features(y, sr)

    return features


@dataclass
class SpeechClassifierOutput(ModelOutput):
    from typing import Optional, Tuple

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Clasificador (Head)"""

    max_lenght = 6828  # cantidad maxima de ventanas en el dataset
    features_size = 151  # cantidad de features acusticos

    def __init__(self, config):
        super().__init__()

        self.pre_mean_wv2 = nn.Linear(in_features=1024, out_features=1024)
        self.pre_mean_features = nn.Linear(
            in_features=self.features_size, out_features=self.features_size
        )

        self.dense = nn.Linear(
            config.hidden_size + self.features_size,
            config.hidden_size + self.features_size,
        )
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(
            config.hidden_size + self.features_size, config.num_labels
        )

    def padded_mean(self, features, mask):
        features = torch.multiply(features, mask)
        mask = torch.sum(mask, 1)
        features = torch.sum(features, 1)
        features = torch.div(features, mask)
        return features

    def forward(self, features, hidden_states, mask, **kwargs):

        features = self.pre_mean_features(features)
        hidden_states = self.pre_mean_wv2(hidden_states)
        # TODO Activacion aca?
        hidden_states = torch.mean(hidden_states, dim=1)
        features = self.padded_mean(features, mask)

        x = torch.cat(
            (features, hidden_states), 1
        )  # x es el vector promediado y concatenado de la salida de w2v2 con los features

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.sigmoid(x)  # TODO chequear otras funciones de activacion
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 para clasificación"""

    def __init__(self, config, class_weights, max_length):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = "mean"
        self.config = config
        self.class_weights = class_weights
        self.max_length = max_length  # 1707 para los features acusticos
        self.init_weights()
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values,
        acoustic_features,
        mask,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        re_d=None,  # return_dict
    ):
        features = acoustic_features
        return_dict = re_d if re_d is not None else self.config.use_return_dict
        input_values = torch.tensor(input_values)
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.classifier(features, hidden_states, mask)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor: The processor used for proccessing the data.
        padding: Select a strategy to pad the returned sequences (according to
            the model's padding side and padding index)
            among:
            * longest: Pad to the longest sequence in the batch
                (or no padding if only a single sequence if provided).
            * max_length: Pad to a maximum length specified with the argument
                max_length` or to the maximum acceptable input length for the
                model if that argument is not provided.
            * False` or do_not_pad (default): No padding
                (i.e., can output a batch with sequences of different lengths).
        max_length: Maximum length of the ``input_values`` of the returned list
            and optionally padding length (see above).
        max_length_labels: Maximum length of the ``labels`` returned list and
            optionally padding length (see above).
        pad_to_multiple_of: If set will pad the sequence to a multiple of the
            provided value. This is especially useful to enable the use of
            Tensor Cores on NVIDIA hardware with compute capability >= 7.5
    """

    processor: Wav2Vec2Processor
    padding = "max_length"
    max_length = 546220
    max_length_labels = None
    pad_to_multiple_of = None
    pad_to_multiple_of_labels = None
    filter = Filter()
    feature_extractor = AcousticFeatureExtractor()
    labels = ["negativo", "positivo"]

    def __call__(self, batch):

        batch = {key: [i[key] for i in batch] for key in batch[0]}
        speech_list = [speech_file_to_array_fn(p) for p in batch["path"]]
        target_list = [
            0 if label == self.labels[0] else 1 for label in batch["emotion"]
        ]

        features = self.processor(speech_list, sampling_rate=16000)

        input_features = self.processor.pad(
            features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        acoustic_features_list = [
            extract_acoustic_features(y, self.feature_extractor, self.filter)
            for y in speech_list
        ]
        mask = []
        for n in range(len(acoustic_features_list)):
            temp_mask = torch.ones(acoustic_features_list[n].shape)
            mask.append(temp_mask)
        acoustic_features_padded = torch.nn.utils.rnn.pad_sequence(
            acoustic_features_list, batch_first=True, padding_value=0.0
        ).to(torch.float32)
        mask = torch.nn.utils.rnn.pad_sequence(
            mask, batch_first=True, padding_value=0.0
        )

        d_type = torch.long if isinstance(target_list[0], int) else torch.float
        input_features["labels"] = torch.tensor(list(target_list), dtype=d_type)
        input_features["acoustic_features"] = acoustic_features_padded
        input_features["mask"] = mask
        # print(torch.cuda.memory_summary())
        return input_features


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)

    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


class CTCTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(
            None,
            None,
        ),
        class_weights=None,
    ):
        self.use_amp = is_apex_available()
        self.class_weights = class_weights
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


class AngerDetector:
    """Clase principal del modelo en su totalidad, contiene la extracción de
    features acústicos y Wav2Vec2.0"""

    labels = ["negativo", "positivo"]
    input_column = "path"
    output_column = "emotion"

    def __init__(
        self,
        base_model: str,
        class_weights=torch.tensor([1, 1], dtype=torch.float32).to("cuda:0"),
        batch_size=8,
        epochs=50,
        save_steps=10,
        eval_steps=10,
    ):
        """
        base_model (str) Nombre del modelo base de wav2vec
        class_weights (list) lista con el peso de las clases (float)

        """
        self.class_weights = class_weights
        self.base_model = base_model
        self.max_length = 1707
        label2id = {label: i for i, label in enumerate(self.labels)}
        id2label = {i: label for i, label in enumerate(self.labels)}

        self.config = AutoConfig.from_pretrained(
            self.base_model,
            num_labels=2,
            label2id=label2id,
            id2label=id2label,
            finetuning_task="wav2vec2_clf",
        )

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            base_model,
        )

        self.sr = self.processor.sampling_rate  # 16k

        self.model = Wav2Vec2ForSpeechClassification.from_pretrained(
            base_model,
            config=self.config,
            class_weights=class_weights,
            max_length=self.max_length,
        )
        self.model.freeze_feature_extractor()

        self.training_args = TrainingArguments(
            output_dir=base_model,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            evaluation_strategy="steps",
            num_train_epochs=epochs,
            # fp16=True,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=10,
            learning_rate=1e-4,
            save_total_limit=2,
            remove_unused_columns=False,
        )
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor)

    def train(self, train_dataset, eval_dataset):
        self.trainer = CTCTrainer(
            model=self.model,
            data_collator=self.data_collator,
            args=self.training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor,
            class_weights=self.class_weights,
        )
        self.trainer.train()


if __name__ == "__main__":
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    import test_training
