from abc import ABC, abstractmethod
import numpy as np
import librosa
from scipy import signal
from librosa.core import stft as STFT
from typing import Tuple


SIGNALS = {
    "hamming": signal.hamming,
    "hann": signal.hann,
    "blackman": signal.blackman,
    "bartlett": signal.bartlett,
}


class SoundAugmentation(ABC):
    """
    Abstract class for sound augmentations.
    """

    @abstractmethod
    def __call__(self, sound: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__


class NoAugmentation(SoundAugmentation):
    def __call__(self, sound: np.ndarray) -> np.ndarray:
        return sound


class Sequential(SoundAugmentation):
    def __init__(self, *augmentations):
        for aug in augmentations:
            if not isinstance(aug, SoundAugmentation):
                raise TypeError(
                    "Expected that augmentation will be inherited "
                    "from 'SoundAugmentation' but got '{}'!".format(type(aug))
                )
        self.augs = augmentations

    def __call__(self, sound):
        for aug in self.augs:
            sound = aug(sound)
        return sound
    
    def __repr__(self):
        return "{name}({params})".format(
            name=self.__class__.__name__, 
            params=",".join([repr(aug) for aug in self.augs])
        )


class OneOf(Sequential):
    def __call__(self, sound):
        return np.random.choice(self.augs)(sound)
    
    def __repr__(self):
        return "{name}({params})".format(
            name=self.__class__.__name__, 
            params=",".join([repr(aug) for aug in self.augs])
        )

class Noise(SoundAugmentation):
    def __init__(self, noise_factor: Tuple[float, float] = (0, 0.05)):
        self.noise_factor = noise_factor

    def __call__(self, sound: np.ndarray) -> np.ndarray:
        noise = np.random.randn(len(sound))
        augmented = (sound + np.random.uniform(*self.noise_factor) * noise).astype(noise.dtype)
        return augmented

    def __repr__(self):
        return f"{self.__class__.__name__}(noise_factor={self.noise_factor})"


class TimeShifting(SoundAugmentation):
    def __init__(self, max_shift: float, direction: str = "both"):
        if direction not in {"both", "left", "right"}:
            raise ValueError("direction should be one of (both, left, right)!")
        self.direction = direction
        self.max_shift = max_shift

    def __call__(self, sound: np.ndarray) -> np.ndarray:
        shift = np.random.randint(int(len(sound) * self.max_shift))
        if self.direction == "right":
            shift = -shift
        elif self.direction == "both":
            shift = np.random.choice((1, -1)) * shift
        augmented = np.roll(sound, shift)
        if shift > 0:
            augmented[:shift] = 0
        else:
            augmented[shift:] = 0
        return augmented

    def __repr__(self):
        return f"{self.__class__.__name__}(max_shift={self.max_shift},direction={self.direction})"


class Pitch(SoundAugmentation):
    def __init__(self, sampling_rate: int, pitch_factor: Tuple[float, float]):
        self.sampling_rate = sampling_rate
        self.pitch_factor = pitch_factor
    
    def __call__(self, sound: np.ndarray) -> np.ndarray:
        return librosa.effects.pitch_shift(
            sound, self.sampling_rate, np.random.uniform(*self.pitch_factor), res_type="kaiser_fast"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(sampling_rate={self.sampling_rate},pitch_factor={self.pitch_factor})"


class TimeStretch(SoundAugmentation):
    def __init__(self, speed_factor: Tuple[float, float]):
        self.speed_factor = speed_factor

    def __call__(self, sound: np.ndarray) -> np.ndarray:
        return librosa.effects.time_stretch(
            sound, np.random.uniform(*self.speed_factor)
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(speed_factor={self.speed_factor})"


class MinMaxScale(SoundAugmentation):
    """
    NOTE: Output sound will be in range [0, 1].
    """
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, sound: np.ndarray) -> np.ndarray:
        _min = np.min(sound)
        _max = np.max(sound)
        return (sound - _min) / (_max - _min + self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.eps})"


class Normalization(SoundAugmentation):
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, sound: np.ndarray) -> np.ndarray:
        inv_std = 1 / (np.std(sound) + self.eps)
        return (sound - np.mean(sound)) * inv_std


class ConstantNormalization(SoundAugmentation):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.inv_std = 1 / std

    def __call__(self, sound: np.ndarray) -> np.ndarray:
        return (sound - self.mean) * self.inv_std


class MelSpectrogram(SoundAugmentation):
    """
    The fastest way to preprocess sound.
    """

    def __init__(self, sample_rate: int, window_size: float, window_stride: float, window: str):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = SIGNALS.get(window, SIGNALS["hamming"])

        self._nfft = int(self.sample_rate * self.window_size)
        self._hop_length = self._nfft
        self._window_length = int(self.sample_rate * self.window_stride)

    def __call__(self, sound: np.ndarray) -> np.ndarray:
        spect = librosa.feature.melspectrogram(
            sound,
            sr=self.sample_rate,
            n_fft=self._nfft,
            win_length=self._window_length,
            hop_length=self._hop_length,
            window=self.window,
        )
        return spect

    def __repr__(self):
        class_repr = "{}(sample_rate={},window_size={},window_stride={},window={})".format(
            self.__class__.__name__,
            self.sample_rate,
            self.window_size,
            self.window_stride,
            self.window,
        )
        return class_repr


class LogMagnitude(SoundAugmentation):
    """
    Second by the speed method for sound preprocessing.
    """

    def __init__(self, sample_rate: int, window_size: float, window_stride: float, window: str):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = SIGNALS.get(window, SIGNALS["hamming"])

        self._nfft = int(self.sample_rate * self.window_size)
        self._hop_length = self._nfft
        self._window_length = int(self.sample_rate * self.window_stride)

    def __call__(self, sound: np.ndarray) -> np.ndarray:
        D = STFT(
            sound, 
            n_fft=self._nfft,
            hop_length=self._hop_length, 
            win_length=self._window_length,
            window=self.window,
        )  # output shapes: (1 + n_fft/2, n_frames)
        spect, phase = librosa.magphase(D)
        spect = np.log1p(spect)
        return spect

    def __repr__(self):
        class_repr = "{}(sample_rate={},window_size={},window_stride={},window={})".format(
            self.__class__.__name__,
            self.sample_rate,
            self.window_size,
            self.window_stride,
            self.window,
        )
        return class_repr


class MFCC(SoundAugmentation):
    """
    Slowest method for sound preprocessing.
    """

    def __init__(self, sample_rate: int, window_size: float, window_stride: float, window: str):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = SIGNALS.get(window, SIGNALS["hamming"])

        self._n_mfcc = int(sample_rate * window_size)
        self._window_length = self._n_mfcc
        self._hop_length = int(sample_rate * window_stride)

    def __call__(self, sound: np.ndarray) -> np.ndarray:
        spect = librosa.feature.mfcc(
            sound,
            sr=self.sample_rate,
            n_mfcc=self._n_mfcc,
            win_length=self._window_length,
            hop_length=self._hop_length,
        )
        return spect

    def __repr__(self):
        class_repr = "{}(sample_rate={},window_size={},window_stride={},window={})".format(
            self.__class__.__name__,
            self.sample_rate,
            self.window_size,
            self.window_stride,
            self.window,
        )
        return class_repr


class FrequencyNormalization(SoundAugmentation):
    """
    NOTE: Should be used only for spectrograms.
    That means that you should use this augmentation after one of:
     - MelSpectrogram
     - LogMagnitude
     - MFCC
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.inv_std = 1 / std
    
    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Expected that sound have shapes: (frequency, time),
        so mean and std should have shape - (frequency,)
        """
        return (spectrogram - self.mean[:, np.newaxis]) * self.inv_std[:, np.newaxis]


__all__ = (
    "SoundAugmentation", 
    "NoAugmentation",
    "Sequential",
    "Noise",
    "TimeShifting",
    "Pitch",
    "TimeStretch",
    "OneOf",
    "MinMaxScale",
    "Normalization",
    "ConstantNormalization",
    "FrequencyNormalization",
    "MelSpectrogram",
    "LogMagnitude",
    "MFCC",
)
