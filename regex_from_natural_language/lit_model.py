import random
import warnings

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .metrics import CER
from .modules import Decoder, Encoder
from .tokenizer import Tokenizer


class LitSeq2Seq(LightningModule):
    def __init__(
        self,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        lr: float,
        weight_decay: float,
        patience: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        teacher_forcing_ratio: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.teacher_forcing_ratio = teacher_forcing_ratio

        src_vocab_size = len(self.src_tokenizer)
        tgt_vocab_size = len(self.tgt_tokenizer)
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)

        self.pad_index = self.src_tokenizer.pad_index
        self.sos_index = self.src_tokenizer.sos_index
        self.eos_index = self.src_tokenizer.eos_index
        self.unk_index = self.src_tokenizer.unk_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_index)
        self.cer = CER(ignore_indices=self.src_tokenizer.ignore_indices)

    def forward(self, src, tgt=None, src_mask=None, teacher_forcing_ratio=None, max_len=None):
        """
        Args:
            src: (src_len, batch_size)
            tgt: (tgt_len, batch_size)

        Returns:
            outputs: (max_len, batch_size, tgt_vocab_size)
            preds: (max_len, batch_size)
        """
        if tgt is None:
            if max_len is None:
                raise ValueError("max_len cannot be None when tgt is None.")
            if teacher_forcing_ratio is not None and teacher_forcing_ratio != 0:
                warnings.warn("No teacher forcing will be used when tgt is None.")
            teacher_forcing_ratio = 0
        else:
            if max_len is None:
                max_len = tgt.size(0)

        batch_size = src.size(1)
        outputs = torch.zeros(max_len, batch_size, self.decoder.tgt_vocab_size, dtype=torch.float32, device=self.device)
        preds = torch.zeros(max_len, batch_size, dtype=torch.int64, device=self.device)
        preds[0] = self.sos_index

        encoder_outputs, hidden = self.encoder(src)
        hidden = hidden[: self.encoder.num_layers]  # Hidden states from encoder's forward RNN
        decoder_input = preds.data[0]
        for t in range(1, max_len):
            output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, src_mask)
            outputs[t] = output.clone()
            top1 = output.argmax(1)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = tgt[t]
            else:
                decoder_input = top1
            preds[t] = top1
        return outputs, preds

    def shared_step(self, batch, mode: str):
        src, tgt, src_mask = batch
        if mode == "train":
            outputs, preds = self(src, tgt, src_mask, teacher_forcing_ratio=self.teacher_forcing_ratio)
        else:
            outputs, preds = self(src, None, src_mask, teacher_forcing_ratio=0, max_len=tgt.size(0))

        outputs = outputs.permute(1, 2, 0)  # (batch_size, tgt_vocab_size, tgt_len)
        preds = preds.permute(1, 0)  # (batch_size, tgt_len)
        tgt = tgt.permute(1, 0)  # (batch_size, tgt_len)
        loss = self.loss_fn(outputs[:, :, 1:], tgt[:, 1:])
        cer = self.cer(preds[:, 1:], tgt[:, 1:])
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}/cer", cer, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.patience, threshold=0.001, mode="min")
        return {
            "optimizer": optimizer,
            "scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
