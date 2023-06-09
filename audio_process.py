from abc import abstractclassmethod, ABC
import librosa
from scipy.signal import lfilter, butter
import numpy as np
from torch import tensor
import Signal_Analysis.features.signal as sig
import torch

# TODO CACHE para optimizar el procesamiento, esta funcionando con cache en RAM
# Hay que ver si sirve o si es mejor pasarlo a disco.
# Tambien falta implementarlo aca. Primero hay que ver por que se llena la VRAM
class AcousticFeatureExtractor:

    win_length = 320
    n_fft = 512
    hop_length = int(win_length // 4)

    def __init__(self) -> None:
        pass

    def get_crest_factor_RMS(self, y, sr):
        s = librosa.stft(
            y=y,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            center=True,
        )
        rms = librosa.feature.rms(S=s, frame_length=self.n_fft)

        # print(frames.shape)
        peaks = max(np.abs(s).tolist())

        crest_factor = np.divide(rms, peaks)
        return crest_factor[0], rms[0]

    def get_mfccs(self, y, sr):
        n_mfcc = 48
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=True,
        )
        mfcc_delta1 = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc_delta1)

        return mfcc.T, mfcc_delta1.T, mfcc_delta2.T

    def get_f0(self, y, sr):
        f0 = librosa.yin(
            y=y,
            fmin=60,
            fmax=350,
            sr=sr,
            frame_length=self.win_length,
            win_length=self.win_length // 2,
            hop_length=self.hop_length,
            center=True,
        )
        f0_delta = librosa.feature.delta(f0)

        return f0, f0_delta

    def get_hnr(self, y, sr):
        # TODO Hacer un codigo para obtener el HNR

        hnr = sig.get_HNR(y, sr, min_pitch=125, periods_per_window=2.5)
        print(len(hnr))
        print(hnr)
        exit()
        # stft_transform = librosa.stft(
        #     y=y,
        #     win_length=self.win_length,
        #     hop_length=self.hop_length,
        #     n_fft=self.n_fft,
        #     center=True,
        # )

        # # Calculate the power spectrum of the signal
        # power_spectrum = np.abs(stft_transform) ** 2

        # power_spectrum = power_spectrum.T
        # harmonic_to_noise = []
        # for window in power_spectrum:
        #     fundamental_frequency = np.argmax(window)
        #     harmonic_frequencies = []
        #     i = 2
        #     harmonic = 0
        #     print(fundamental_frequency)
        #     while harmonic < 257:
        #         i = i + 1
        #         harmonic = fundamental_frequency * i
        #         print(harmonic)
        #         if harmonic < 257:
        #             harmonic_frequencies.append(harmonic)
        #     harmonic_power = [window[freq] for freq in harmonic_frequencies]
        #     harmonic_power = sum(harmonic_power)
        #     noise_power = sum(window) - harmonic_power
        #     hnr = harmonic_power / noise_power
        #     harmonic_to_noise.append(hnr)

        # return harmonic_to_noise

    def get_spectral_centroid(self, y, sr):
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            win_length=self.win_length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True,
        )
        return spectral_centroid[0]

    def get_spectral_rollof(self, y, sr):
        spectral_rollof = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            win_length=self.win_length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True,
        )
        return spectral_rollof[0]

    def get_zero_crossing_rate(self, y, sr):
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=y,
            frame_length=self.win_length,
            hop_length=self.hop_length,
            center=True,
        )
        return zero_crossing_rate[0]

    def extract_features(self, y, sr):
        crest_factor, rms = self.get_crest_factor_RMS(y, sr)
        mfcc, mfcc_delta1, mfcc_delta2 = self.get_mfccs(y, sr)
        f0, f0_delta = self.get_f0(y, sr)
        # hnr = self.get_hnr(y, sr)
        spectral_centroid = self.get_spectral_centroid(y, sr)
        spectral_rollof = self.get_spectral_rollof(y, sr)
        zero_crossing_rate = self.get_zero_crossing_rate(y, sr)

        return torch.transpose(
            tensor(
                np.concatenate(
                    (
                        tensor(np.expand_dims(crest_factor, axis=0)),
                        tensor(np.expand_dims(rms, axis=0)),
                        tensor(mfcc.T),
                        tensor(mfcc_delta1.T),
                        tensor(mfcc_delta2.T),
                        tensor(np.expand_dims(f0, axis=0)),
                        tensor(np.expand_dims(f0_delta, axis=0)),
                        # tensor(np.expand_dims(hnr, axis=0)),
                        tensor(np.expand_dims(spectral_centroid, axis=0)),
                        tensor(np.expand_dims(spectral_rollof, axis=0)),
                        tensor(np.expand_dims(zero_crossing_rate, axis=0)),
                    ),
                    axis=0,
                )
            ),
            0,
            1,
        )


class Processor(ABC):
    @abstractclassmethod
    def process(self, audio, fs):
        pass


class Filter(Processor):
    def __init__(self, order=5):
        self.order = order

    def process(self, audio, fs):
        b, a = self.butter_params(fs)
        y = lfilter(b, a, audio)
        return y

    def butter_params(self, fs):
        nyq = 0.5 * fs
        low_freq = 300
        high_freq = 3300
        low = low_freq / nyq
        high = high_freq / nyq
        b, a = butter(self.order, [low, high], btype="band")
        return b, a
