import os
import pydub
import librosa
import numpy as np
from scipy.io import wavfile
from typing import List, Tuple
from tempfile import NamedTemporaryFile


_NORMALIZATION_CONST = 2 ** 15
KNOWN_FORMATS = {
    "wav"
}


def read_wav(f, sample_rate: int = None) -> np.ndarray:
    sound, _ = librosa.core.load(f, sr=sample_rate)
    sound = sound[np.isfinite(sound)]
    return sound


def read_as_wav(f) -> np.ndarray:
    extension = f[-3:]
    audio = pydub.AudioSegment.from_file(f, format=extension)
    audio.set_frame_rate(16_000)
    audio.set_channels(1)
    
    with NamedTemporaryFile(suffix=".wav") as tmp_file:
        audio.export(tmp_file.name, bitrate="16k", format="wav")
        data = read_wav(tmp_file.name)

    return data


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = f"sox \"{path}\" -r {sample_rate} -c 1 -b 16 -e si {tar_filename} trim {start_time} ={end_time} >/dev/null 2>&1"
        os.system(sox_params)
        y = read_wav(tar_filename)
        return y


def augment_audio_with_sox(path, 
                           sample_rate, 
                           tempo, 
                           gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_augment_params = " ".join(sox_augment_params)
        sox_params = f"sox \"{path}\" -r {sample_rate} -c 1 -b 16 -e si {augmented_filename} {sox_augment_params} >/dev/null 2>&1"
        os.system(sox_params)
        y = read_wav(augmented_filename)
        return y


def load_randomly_augmented_audio(path, 
                                  sample_rate=16000, 
                                  tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(
        low=low_tempo, 
        high=high_tempo
    )
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(
        low=low_gain, 
        high=high_gain
    )
    audio = augment_audio_with_sox(
        path=path, 
        sample_rate=sample_rate, 
        tempo=tempo_value, 
        gain=gain_value
    )
    return audio


def read_text(fname: str) -> str:
    with open(str(fname), "r") as f:
        content = f.read().strip()
    return content


def one_hot_encode(items: List[int], num_values: int = None, dtype="float32") -> np.ndarray:
    if num_values is None:
        num_values = max(items)
    res = np.zeros((len(items), num_values), dtype=dtype)
    for row, col in enumerate(items):
        res[row, col] = 1
    return res

