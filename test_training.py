from datasets import load_dataset
import atexit
import os

# Script hecho para debuggear el entrenamiento


@atexit.register
def remove_files():
    os.system("rm -r facebook")


save_path = "csv_files"

data_files = {
    "train": f"{save_path}/train_minimal_100.csv",
    "validation": f"{save_path}/test_minimal_100.csv",
}

dataset = load_dataset(
    "csv",
    data_files=data_files,
    delimiter="\t",
)
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

input_column = "path"
output_column = "emotion"

from model import AngerDetector

anger_detector = AngerDetector(
    base_model="facebook/wav2vec2-large-xlsr-53", epochs=1, batch_size=1, eval_steps=1
)
anger_detector.train(train_dataset, eval_dataset)
