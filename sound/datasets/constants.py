import json
import numpy as np
from typing import List, Tuple, Union


class CharacterTokenizer:
    def __init__(self, 
                 vocab: List[str],
                 is_lowercase: bool = False,
                 blank_token: str = "."):
        
        if blank_token in vocab:
            raise ValueError("Blank token should be a unique token and not presented in vocab!")

        self.index_to_token = [blank_token] + vocab
        self.token_to_index = {c: idx for idx, c in enumerate(self.index_to_token)}

        self.blank_token = blank_token
        self.eow_token_id = self.token_to_index[blank_token]
        
        self.is_lowercase = bool(is_lowercase)
        self.num_tokens = len(self.token_to_index)

    def to_token_ids(self, text: str) -> List[int]:
        text = text.lower() if self.is_lowercase else text
        # convert '\t', '\n' and other space tokens to ' '
        text = " ".join(text.split())
        text = filter(lambda letter: letter in self.token_to_index and letter != self.blank_token, text)
        return [self.token_to_index[c] for c in text]

    def to_text(self, tokens: List[int]) -> str:
        text = "".join([self.index_to_token[t] for t in tokens if 0 <= t < self.num_tokens])
        return text

    @staticmethod
    def from_file(file: str):
        with open(file, "r") as f:
            content = json.load(f)
        return CharacterTokenizer(**content)


# '_' - space token
TOKENIZER = CharacterTokenizer(
    vocab=[
        " ","а","б","в","г","д","е","ж","з",
        "и","й","к","л","м","н","о","п","р",
        "с","т","у","ф","х","ц","ч","ш","щ",
        "ъ","ы","ь","э","ю","я","ё",
    ],
    blank_token=".",
    is_lowercase=True,
)


if __name__ == '__main__':
    print(TOKENIZER.token_to_index)
    print(TOKENIZER.num_tokens)
    print(TOKENIZER.to_token_ids("привет как дела"))