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
from transformers import EvalPrediction, TrainingArguments


@dataclass
class SpeechClassifierOutput(ModelOutput):
    from typing import Optional, Tuple

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Clasificador (Head)"""

    max_lenght = 1707

    def __init__(self, config):
        super().__init__()
        # TODO Definir modelo
        self.pre_mean_wv2 = nn.Linear(self.max_lenght)
        self.pre_mean_features = nn.Linear(self.max_lenght)

        # TODO de qué tamaño quedaría esto?
        self.dense = nn.Linear(
            config.hidden_size + 57,
            config.hidden_size + 57,
        )
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size + 57, config.num_labels)

    def forward(self, features, hidden_states, mask, **kwargs):
        features = self.pre_mean_features(features)
        hidden_states = self.pre_mean_wv2(hidden_states)
        # TODO Aplicar mascara
        features = torch.mean(features, dim=1)
        hidden_states = torch.mean(hidden_states, dim=1)
        x = torch.cat((features, hidden_states), 1)
        # x es el vector promediado y concatenado de la salida de wv2
        # y los parametroos acusticos
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 para clasificación"""

    def __init__(self, config, class_weights, max_lenght):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = "mean"
        self.config = config
        self.class_weights = class_weights
        self.max_length = max_lenght  # 1707 para los features acusticos
        self.init_weights()
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        # TODO Averiguar bien qué es esto, esto sería el promedio?
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                """The pooling method hasn't been defined! Your pooling mode
                must be one of these ['mean', 'sum', 'max']"""
            )

        return outputs

    def forward(
        self,
        input_values,
        features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        re_d=None,  # return_dict
    ):
        return_dict = re_d if re_d is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        features = torch.tensor(features)
        hidden_mask = []
        for n, hidden_state in enumerate(hidden_states):
            l_feature = len(hidden_state)
            ones = torch.ones(l_feature)
            zeros = torch.zeros(self.max_length - l_feature)
            feature = torch.cat((hidden_state, zeros), 1)
            hidden_states[n] = feature
            mask = torch.cat((ones, zeros), 1)
            hidden_mask.append(mask)
        mask = []
        for n, feature in enumerate(features):
            l_feature = len(feature)
            ones = torch.ones(l_feature)
            zeros = torch.zeros(self.max_length - l_feature)
            feature = torch.cat((feature, zeros), 1)
            features[n] = feature
            feature_mask = torch.cat((ones, zeros), 1)
            mask.append(feature_mask)

        mask = torch.cat(tuple(mask), 0)  # las mascaras deberian ser iguales
        # size = features.size()
        # TODO configurar
        # features = torch.reshape(features, (size[0], size[-1]))
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

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [feature["labels"] for feature in features]
        acoustic_features = [feature["features"] for feature in features]
        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        # TODO imprimir para ver el tamaño de la ventana que esta usando
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # batch = self.processor.pad(
        #     acoustic_features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors="pt",
        # )
        batch["labels"] = torch.tensor(label_features, dtype=d_type)
        batch["features"] = torch.tensor(acoustic_features)
        return batch


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)

    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


class AngerDetector:
    """Clase principal del modelo en su totalidad, contiene la extracción de
    features acústicos y Wav2Vec2.0"""

    labels = ["negativo", "positivo"]
    input_column = "path"
    output_column = "emotion"

    def __init__(
        self,
        base_model: str,
        class_weights: list = [1, 1],
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
        )
        self.model.freeze_feature_extractor()

        training_args = TrainingArguments(
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
        )

    def speech_file_to_array_fn(self, path):
        """Carga un audio y lo resamplea al sample rate del modelo"""

        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.sr)

        speech = resampler(speech_array).squeeze().numpy()

        return speech

    def extract_acoustic_features(self, y):
        """Extrae los features acústicos del audio"""

        filter = Filter()
        y = filter.process(y, self.sr)
        feature_extractor = AcousticFeatureExtractor()
        features = feature_extractor(y, self.sr)

        return features

    def preprocess_function(self, batch):
        """arma el objeto que ingresa en el modelo
        batch es una Dataset con paths a audios y su etiqueta"""

        speech_list = [
            self.speech_file_to_array_fn(p) for p in batch[self.input_column]
        ]

        target_list = [
            0 if label == self.label_list[0] else 1
            for label in batch[self.output_column]
        ]

        acoustic_features_list = [
            self.extract_acoustic_features(y) for y in speech_list
        ]

        # TODO Chequear el tema del padding

        result = self.processor(speech_list, sampling_rate=self.sr)
        result["labels"] = list(target_list)
        result["speech_list"] = speech_list
        result["acoustic features"] = acoustic_features_list
        return result
