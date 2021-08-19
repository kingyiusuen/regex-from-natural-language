import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class Tokenizer:
    def __init__(self, token_to_index: Optional[Dict[str, int]] = None) -> None:
        self.token_to_index: Dict[str, int]
        self.index_to_token: Dict[int, str]

        if token_to_index:
            self.token_to_index = token_to_index
            self.index_to_token = {index: token for token, index in self.token_to_index.items()}
        else:
            self.token_to_index = {}
            self.index_to_token = {}
            self._add_token("<PAD>")
            self._add_token("<SOS>")
            self._add_token("<EOS>")
            self._add_token("<UNK>")

        self.pad_index = self.token_to_index["<PAD>"]
        self.sos_index = self.token_to_index["<SOS>"]
        self.eos_index = self.token_to_index["<EOS>"]
        self.unk_index = self.token_to_index["<UNK>"]
        self.ignore_indices = set([self.pad_index, self.sos_index, self.eos_index, self.unk_index])

    def _add_token(self, token: str) -> int:
        """Add one token to the vocabulary.

        Args:
            token: The token to be added.

        Returns:
            The index of the input token.
        """
        if token in self.token_to_index:
            return self.token_to_index[token]
        index = len(self)
        self.token_to_index[token] = index
        self.index_to_token[index] = token
        return index

    def __len__(self):
        return len(self.token_to_index)

    def train(self, texts: List[List[str]], min_count: int = 2) -> None:
        """Create a mapping from tokens to indices and vice versa."""
        # Count the frequency of each token
        counter: Dict[str, int] = {}
        for text in texts:
            for token in text:
                counter[token] = counter.get(token, 0) + 1

        for token, count in counter.items():
            # Remove tokens that show up fewer than `min_count` times
            if count < min_count:
                continue
            index = len(self)
            self.index_to_token[index] = token
            self.token_to_index[token] = index

    def encode(self, formula: List[str]) -> List[int]:
        indices = [self.sos_index]
        for token in formula:
            index = self.token_to_index.get(token, self.unk_index)
            indices.append(index)
        indices.append(self.eos_index)
        return indices

    def decode(self, indices: List[int], inference: bool = True) -> List[str]:
        tokens = []
        for index in indices:
            if index not in self.index_to_token:
                raise RuntimeError(f"Found an unknown index {index}")
            if index == self.eos_index:
                break
            if inference and index in self.ignore_indices:
                continue
            token = self.index_to_token[index]
            tokens.append(token)
        return tokens

    def save(self, filename: Union[Path, str]):
        with open(filename, "w") as f:
            json.dump(self.token_to_index, f)

    @classmethod
    def load(cls, filename: Union[Path, str]) -> "Tokenizer":
        with open(filename) as f:
            token_to_index = json.load(f)
        return cls(token_to_index)
