from audio_process import AcousticFeatureExtractor, Filter
import torchaudio
import glob
import random
import time
from functools import wraps

cache_dict = {}


def cache(func):
    @wraps(func)
    def check_if_in_cache(*args, **kwargs):

        if kwargs:
            y = args[0]
            path = kwargs["path"]
            path = path + "_features"

        else:
            path = args[0]

        if path in cache_dict:
            return cache_dict[path]
        else:
            if kwargs:
                result = func(y, path)
                cache_dict[path] = result
            else:
                result = func(path)
                cache_dict[path] = result
            return result

    return check_if_in_cache


@cache
def speech_file_to_array_fn(path=None):
    """Carga un audio y lo resamplea al sample rate del modelo"""

    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)

    speech = resampler(speech_array).squeeze().numpy()

    return speech


@cache
def extract_acoustic_features(y, path=None):
    """filtra el audio y extrae los features acústicos del audio"""

    filter = Filter()
    y = filter.process(y, 16000)
    feature_extractor = AcousticFeatureExtractor()
    features = feature_extractor.extract_features(y, 16000)

    return features


def random_example(directory=None, k=None, files=None):
    "Carga k audios aleatorio del directorio"
    if not directory and not files:
        raise ("Error: no se ingresó ni directorio ni archivos")
    if directory:
        files = glob.glob(directory)
    if k:
        examples = random.choices(files, k=k)
    else:
        examples = files
    features = []
    t1 = time.time()
    for example in examples:
        y = speech_file_to_array_fn(example)
        feature = extract_acoustic_features(y, path=example)
        features.append(feature)
    t2 = time.time()
    if k:
        print(
            f"""FINALIZADO.\n
            Cantidad de audios: {k}\n
            Tiempo total: {round(t2-t1,3)}\n
            Tiempo promedio:{round(((t2-t1)/k),3)}"""
        )
    return features


def random_example_cache(directory, k, n):
    files = glob.glob(directory)
    if k:
        examples = random.choices(files, k=k)
    else:
        examples = files
    for i in range(n):
        t1 = time.time()
        _ = random_example(files=examples)
        print(_)
        t2 = time.time()
        print(round(t2 - t1, 3))


if __name__ == "__main__":
    "Prueba con 1000 ejemplos aleatorios"
    directory = "/home/franco/tesis/IEMOCAP_mp3/*.mp3"
    # features = random_example(directory, k=1)
    features = random_example_cache(directory, 100, 10)
