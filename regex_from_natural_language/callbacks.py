from pytorch_lightning.callbacks.base import Callback


class PredictionWriter(Callback):
    def __init__(self, filename: str = "predictions"):
        super().__init__()
        self.filename = filename

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        with open(f"{self.filename}.txt", "a") as f:
            for pred in outputs["preds"]:
                tokens = pl_module.tgt_tokenizer.decode(pred.tolist())
                f.write(" ".join(tokens) + "\n")
