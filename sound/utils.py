import pandas as ps
from copy import copy
from pathlib import Path
from sklearn.model_selection import train_test_split


DEFAULT_TRAIN_VALID_SPLIT = 0.25
DEFAULT_RANDOM_STATE = 2020


def _make_mp3_file(txt_file: str) -> str:
    if txt_file.endswith(".txt"):
        return txt_file[:-4] + ".mp3"
    return txt_file


class DataStorage:
    def __init__(self):
        self.data_dir = None
        self.train_files = None
        self.valid_files = None
        self.random_state = DEFAULT_RANDOM_STATE
        self.train_valid_split = DEFAULT_TRAIN_VALID_SPLIT

    def _read_files(self) -> list:
        files = [
            (_make_mp3_file(str(txt_file)), str(txt_file))  # (sound file, txt file)
            for txt_file in Path(self.data_dir).glob("**/*.txt")
        ]
        return files

    def get_files(self, 
                  data_dir: str, 
                  train_valid_split: float = None, 
                  random_state: int = None) -> tuple:
        """
        Load data (if not loaded) and split to train & validation sets and return them,
        if it was already made then return cached data.
        """
        random_state = DEFAULT_RANDOM_STATE or random_state
        train_valid_split = DEFAULT_TRAIN_VALID_SPLIT or train_valid_split

        if self.data_dir is None or self.data_dir != data_dir or \
            self.random_state != random_state or \
            self.train_valid_split != train_valid_split:
            # new directory or changed split parameters -> reload content
            self.data_dir = data_dir
            self.train_files = None
            self.valid_files = None
            self.random_state = random_state
            self.train_valid_split = train_valid_split

        if self.train_files is None and self.valid_files is None:
            files = self._read_files()
            self.train_files, self.valid_files = train_test_split(
                files, 
                test_size=self.train_valid_split, 
                random_state=self.random_state, 
                shuffle=True
            )
        return copy(self.train_files), copy(self.valid_files)


    def get_dfs(self,
                csv_file: str,
                train_valid_split: float = None, 
                random_state: int = None) -> tuple:

        random_state = DEFAULT_RANDOM_STATE or random_state
        train_valid_split = DEFAULT_TRAIN_VALID_SPLIT or train_valid_split

        if self.data_dir is None or self.data_dir != csv_file or \
            self.random_state != random_state or \
            self.train_valid_split != train_valid_split:
            # new directory or changed split parameters -> reload content
            self.data_dir = csv_file
            self.train_files = None
            self.valid_files = None
            self.random_state = random_state
            self.train_valid_split = train_valid_split

        if self.train_files is None and self.valid_files is None:
            data = ps.read_csv(self.data_dir)
            self.train_files, self.valid_files = train_test_split(
                data, 
                test_size=self.train_valid_split, 
                random_state=self.random_state, 
                shuffle=True
            )
        return copy(self.train_files), copy(self.valid_files)

