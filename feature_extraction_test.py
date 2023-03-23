from audio_process import AcousticFeatureExtractor, Filter
import torchaudio
import glob
import random
import time


def speech_file_to_array_fn(path):
    """Carga un audio y lo resamplea al sample rate del modelo"""

    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)

    speech = resampler(speech_array).squeeze().numpy()

    return speech


def extract_acoustic_features(y):
    """filtra el audio y extrae los features acústicos del audio"""

    filter = Filter()
    y = filter.process(y, 16000)
    feature_extractor = AcousticFeatureExtractor()
    features = feature_extractor.extract_features(y, 16000)

    return features


def random_example(directory, k=1):
    "Carga k audios aleatorio del directorio"
    files = glob.glob(directory)
    examples = random.choices(files, k=k)
    features = []
    t1 = time.time()
    for example in examples:
        y = speech_file_to_array_fn(example)
        feature = extract_acoustic_features(y)
        features.append(feature)
    t2 = time.time()

    print(
        f"""FINALIZADO.\n
        Cantidad de audios: {k}\n
        Tiempo total: {round(t2-t1,3)}\n
        Tiempo promedio:{round(((t2-t1)/k),3)}"""
    )
    return features


if __name__ == "__main__":
    "Prueba con 1000 ejemplos aleatorios"
    directory = "/home/franco/tesis/IEMOCAP_mp3/*.mp3"
    features = random_example(directory, k=10)
