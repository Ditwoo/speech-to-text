import os
import librosa
import torch
import soundfile
import numpy as np
from os.path import join
from copy import copy
from pandas import DataFrame
from typing import List, Tuple, Any, Callable
from torch.utils.data import Dataset

# from scipy.signal import stft as STFT
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .utils import (
    read_wav, 
    read_as_wav,
    read_text, 
    one_hot_encode, 
    load_randomly_augmented_audio,
)
from .constants import TOKENIZER
from .augmentations import SoundAugmentation


class SoundDataset(Dataset):
    def __init__(self,
                 sound_files: List[str],
                 data_dir: str = "",
                 target: List[str] = None,
                 # sound parameters
                 read_as_wav_: bool = False,
                 augmentations: Callable = None):
        """
        :param sound_files: list of files to use for training
        :param data_dir: directory with sound files
        :param target: list of strings (text on sound records)
        :param read_as_waw_: bool, should be true if data is not in wav format
        :param augmentations: Callable, sound augmentations
        """
        self.reader_foo = read_as_wav if read_as_wav_ else read_wav
        self.augmentations: Callable = augmentations

        self.data_dir: str = data_dir
        self.sound_files: List[str] = sound_files
        self.target: List[str] = target

    def __len__(self) -> int:
        return len(self.sound_files)

    def _getitem(self, index: int) -> dict:
        res = dict()
        # get sound from somewhere
        sound_file = self.sound_files[index] if not self.data_dir else join(self.data_dir, self.sound_files[index])
        sound = self.reader_foo(sound_file)
        # prepare audio for model
        if self.augmentations is not None:
            sound = self.augmentations(sound)
        res["features"] = torch.from_numpy(sound)
        # make targets if specified
        if self.target is not None:
            text: str = self.target[index]
            text_token_ids: List[int] = TOKENIZER.to_token_ids(text)
            # num_tokens = len(text_token_ids)
            res["targets"] = text_token_ids
            res["text"] = text
        return res

    def __getitem__(self, index: int) -> dict:
        try:
            record_content = self._getitem(index)
        except Exception as e:
            print(f"Something wrong with file {self.sound_files[index]}!", flush=True)
            raise e
        return record_content


class DFDataset(SoundDataset):
    def __init__(self, 
                 files: DataFrame,
                 data_dir: str = "",
                 sound_files_col: str = "files",
                 text_files_col: str = "text",
                 is_train: bool = True,
                 # sound parameters
                 augmentations: SoundAugmentation = None,
                 read_as_wav_: bool = False):
        sound_files = [files.at[idx, sound_files_col] for idx in files.index]
        if is_train:
            text_files = [files.at[idx, text_files_col] for idx in files.index]
            if data_dir:  # not "" or None
                text_files = [join(data_dir, f) for f in text_files]
            texts = Parallel(n_jobs=os.cpu_count())(delayed(read_text)(f) for f in text_files)
        else:
            texts = None
        super().__init__(
            sound_files=sound_files,
            data_dir=data_dir,
            target=texts,
            read_as_wav_=read_as_wav_,
            augmentations=augmentations,
        )


class CTCCollateFn:
    """
    Generating tensors from batch for CTC loss.

    NOTE: If `target_name` is None then collate function will work in test mode (otput only features).
    """

    def __init__(self, 
                 features_name: str,
                 target_name: str = None,
                 
                 device: str = "cuda"):
        self.features_name = features_name
        self.target_name = target_name
        self.device = torch.device(device)
        self.model_output_lenghts = int(os.environ.get("MODEL_OUTPUTS_LENGTH", 0))

    def sound_seq_len(self, elems: dict) -> int:
        return elems[self.features_name].size(1)

    def _test_output(self, batch: list) -> dict:
        batch_size = len(batch)
        longest_sample = max(batch, key=self.sound_seq_len)[self.features_name]
        freq_size = longest_sample.shape[0]
        max_seq_len = longest_sample.shape[1]
        inputs = torch.zeros(batch_size, freq_size, max_seq_len)
        for idx, sample in enumerate(batch):
            feats = sample[self.features_name]
            seq_len = feats.size(1)
            inputs[idx].narrow(1, 0, seq_len).copy_(feats)
        return {"features": inputs}

    def _train_output(self, batch: list) -> dict:

        # sort by longest sequence of frames
        batch = sorted(batch, key=self.sound_seq_len, reverse=True)
        longest_sample = max(batch, key=self.sound_seq_len)[self.features_name]
        freq_size = longest_sample.shape[0]

        batch_size = len(batch)
        max_seq_len = longest_sample.shape[1]
        max_target_len = max(len(elems[self.target_name]) for elems in batch)

        inputs = torch.zeros(batch_size, freq_size, max_seq_len)
        # input_percentages = torch.FloatTensor(batch_size)
        target_sizes = torch.IntTensor(batch_size)
        input_sizes = torch.IntTensor(batch_size)

        targets = np.zeros((batch_size, max_target_len), dtype=int)
        texts = [""] * batch_size
        for idx, sample in enumerate(batch):
            feats = sample[self.features_name]
            target = sample[self.target_name]
            seq_len = feats.size(1)  # use `n_frames` as sequence length
            inputs[idx].narrow(1, 0, seq_len).copy_(feats)
            targets[idx, :len(target)] = target
            input_sizes[idx] = seq_len
            target_sizes[idx] = len(target)
            texts[idx] = sample["text"]

        targets = torch.from_numpy(targets)

        res = {
            # model input features
            "features": inputs,
            # "lengths": input_sizes,
            # loss features
            "targets": targets, 
            # "input_lengths": input_sizes,
            "target_lengths": target_sizes,
            # for callbacks
            "texts": texts,
        }
        if self.model_output_lenghts:
            res["lengths"] = input_sizes
        else:
            res["input_lengths"] = input_sizes

        return res

    def __call__(self, *args, **kwargs) -> dict:
        if self.target_name is None:
            return self._test_output(*args, **kwargs)
        else:
            return self._train_output(*args, **kwargs)
