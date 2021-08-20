import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .tokenizer import Tokenizer


URLS = {
    "src-train.txt": "https://raw.githubusercontent.com/nicholaslocascio/deep-regex/master/deep-regex-model/data_turk_eval_turk/src-train.txt",  # noqa: E501
    "src-val.txt": "https://raw.githubusercontent.com/nicholaslocascio/deep-regex/master/deep-regex-model/data_turk_eval_turk/src-val.txt",  # noqa: E501
    "src-test.txt": "https://raw.githubusercontent.com/nicholaslocascio/deep-regex/master/deep-regex-model/data_turk_eval_turk/src-test.txt",  # noqa: E501
    "targ-train.txt": "https://raw.githubusercontent.com/nicholaslocascio/deep-regex/master/deep-regex-model/data_turk_eval_turk/targ-train.txt",  # noqa: E501
    "targ-val.txt": "https://raw.githubusercontent.com/nicholaslocascio/deep-regex/master/deep-regex-model/data_turk_eval_turk/targ-val.txt",  # noqa: E501
    "targ-test.txt": "https://raw.githubusercontent.com/nicholaslocascio/deep-regex/master/deep-regex-model/data_turk_eval_turk/targ-test.txt",  # noqa: E501
}


def read_txt(filename):
    return [line.strip().split(" ") for line in open(filename)]


class NLRXDataset(Dataset):
    def __init__(
        self,
        data_dirname: Path,
        src_file: str,
        tgt_file: str,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
    ):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src = read_txt(data_dirname / src_file)
        self.tgt = read_txt(data_dirname / tgt_file)

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx) -> Tuple[List[int], List[int]]:
        src = self.src_tokenizer.encode(self.src[idx])
        tgt = self.tgt_tokenizer.encode(self.tgt[idx])
        return src, tgt


class NLRXDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers, pin_memory):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @property
    def data_dirname(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    @property
    def src_tokenizer_filename(self) -> Path:
        return self.data_dirname / "src_tokenizer.json"

    @property
    def tgt_tokenizer_filename(self) -> Path:
        return self.data_dirname / "tgt_tokenizer.json"

    def prepare_data(self) -> None:
        """Download data and build vocabularies."""
        curr_dir = os.getcwd()
        self.data_dirname.mkdir(parents=True, exist_ok=True)
        os.chdir(self.data_dirname)

        for filename, url in URLS.items():
            if not Path(filename).is_file():
                print(f"Downloading {filename}...")
                os.system(f"curl -s '{url}' -o {filename}")

        if not self.src_tokenizer_filename.is_file():
            print("Building vocabularies for source...")
            train_src = read_txt("src-train.txt")
            src_tokenizer = Tokenizer()
            src_tokenizer.train(train_src)
            src_tokenizer.save(self.src_tokenizer_filename)

        if not self.tgt_tokenizer_filename.is_file():
            print("Building vocabularies for target...")
            train_tgt = read_txt("targ-train.txt")
            tgt_tokenizer = Tokenizer()
            tgt_tokenizer.train(train_tgt)
            tgt_tokenizer.save(self.tgt_tokenizer_filename)

        os.chdir(curr_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        self.src_tokenizer = Tokenizer.load(self.src_tokenizer_filename)
        self.tgt_tokenizer = Tokenizer.load(self.tgt_tokenizer_filename)

        if stage in ("fit", None):
            self.train_dataset = NLRXDataset(
                self.data_dirname, "src-train.txt", "targ-train.txt", self.src_tokenizer, self.tgt_tokenizer
            )
            self.val_dataset = NLRXDataset(
                self.data_dirname, "src-val.txt", "targ-val.txt", self.src_tokenizer, self.tgt_tokenizer
            )

        if stage in ("test", None):
            self.test_dataset = NLRXDataset(
                self.data_dirname, "src-test.txt", "targ-test.txt", self.src_tokenizer, self.tgt_tokenizer
            )

    def collate_fn(self, batch):
        srcs, tgts = zip(*batch)
        assert len(srcs) == len(tgts)
        batch_size = len(srcs)
        src_len = max(len(src) for src in srcs)
        tgt_len = max(len(tgt) for tgt in tgts)
        srcs_tensor = torch.full((src_len, batch_size), fill_value=self.src_tokenizer.pad_index)
        tgts_tensor = torch.full((tgt_len, batch_size), fill_value=self.src_tokenizer.pad_index)
        src_mask = torch.zeros((src_len, batch_size), dtype=torch.bool)
        for i, (src, tgt) in enumerate(zip(srcs, tgts)):
            srcs_tensor[: len(src), i] = torch.tensor(src)
            tgts_tensor[: len(tgt), i] = torch.tensor(tgt)
            src_mask[: len(src), i] = True
        return srcs_tensor, tgts_tensor, src_mask

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
