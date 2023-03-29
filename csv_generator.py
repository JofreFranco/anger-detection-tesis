import torchaudio
from tqdm import tqdm
from pathlib import Path
from pandas import DataFrame
from sklearn.model_selection import train_test_split

data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
save_path = "csv_files"
for path in tqdm(Path("/home/franco/tesis/IEMOCAP_mp3").glob("*.mp3")):
    name = str(path).split("/")[-1].split(".")[0]
    emotion = name.split("_")[-1]
    if emotion == "ang" or emotion == "fru":
        label = "positivo"
    else:
        label = "negativo"
    sesion = name.split("_")[0][3:-1]
    match sesion:  # leave one session out
        case "01":
            data1.append({"name": name, "path": path, "emotion": label})
        case "02":
            data2.append({"name": name, "path": path, "emotion": label})
        case "03":
            data3.append({"name": name, "path": path, "emotion": label})
        case "04":
            data4.append({"name": name, "path": path, "emotion": label})
        case "05":
            data5.append({"name": name, "path": path, "emotion": label})

data = [data1, data2, data3, data4, data5]

for n in range(1, 5):
    test_data = data[n - 1]
    train_data = []
    for i, sesion in enumerate(data):
        if i != n - 1:
            train_data += sesion
    train_df = DataFrame(train_data)
    test_df = DataFrame(test_data)
    train_df.to_csv(
        f"{save_path}/train{n}.csv", sep="\t", encoding="utf-8", index=False
    )
    test_df.to_csv(f"{save_path}/test{n}.csv", sep="\t", encoding="utf-8", index=False)

# datasets solo para debugear
minimal_set_100 = DataFrame(data1[:100])
train_df, test_df = train_test_split(
    minimal_set_100,
    test_size=0.2,
    random_state=101,
    stratify=minimal_set_100["emotion"],
)
train_df.to_csv(
    f"{save_path}/train_minimal_100.csv", sep="\t", encoding="utf-8", index=False
)
test_df.to_csv(
    f"{save_path}/test_minimal_100.csv", sep="\t", encoding="utf-8", index=False
)
