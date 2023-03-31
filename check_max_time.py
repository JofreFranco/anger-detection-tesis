import math
import glob
import numpy as np
import librosa
import torchaudio
from audio_process import AcousticFeatureExtractor, Filter
import torch


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


files = glob.glob("/home/franco/tesis/IEMOCAP_mp3/*.mp3")
L = 0
window_amounts = []
win_length = 320
n_fft = 512
hop_length = int(win_length // 4)
for file in files:
    speech_array, sampling_rate = torchaudio.load(file)
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, 16000)
    s = librosa.stft(
        y=speech_array,
        win_length=win_length,
        hop_length=hop_length,
        n_fft=n_fft,
        center=True,
    )
    window_amounts.append((s.shape[1]))
print(max(window_amounts))
