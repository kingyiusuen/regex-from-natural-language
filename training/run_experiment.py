from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from regex_from_natural_language import LitSeq2Seq, NLRXDataModule
from regex_from_natural_language.callbacks import PredictionWriter


def _setup_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.99)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    return parser


def main() -> None:
    parser = _setup_parser()
    args = parser.parse_args()

    seed_everything(args.seed)

    datamodule = NLRXDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.gpus is not None and args.gpus != 0),
    )
    datamodule.prepare_data()
    datamodule.setup()

    model = LitSeq2Seq(
        src_tokenizer=datamodule.src_tokenizer,
        tgt_tokenizer=datamodule.tgt_tokenizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        factor=args.factor,
        patience=args.patience,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
    )

    model_checkpoint_callback = ModelCheckpoint(filename="best_model", monitor="val/loss", mode="min")
    prediction_writer_callback = PredictionWriter(filename="predictions")
    callbacks = [model_checkpoint_callback, prediction_writer_callback]

    logger = WandbLogger(name="regex-from-natural-language", log_model=True)
    logger.watch(model)
    logger.log_hyperparams(vars(args))

    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
