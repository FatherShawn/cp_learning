from cp_flatten import QuackConstants
from cp_tokenized_data import QuackTokenizedDataModule
from autoencoder import QuackAutoEncoder
from pytorch_lightning import Trainer
from argparse import ArgumentParser


def main() -> None:
    # Add args to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str, default='/data')
    arg_parser.add_argument('--batch_size', type=int, default=4)
    arg_parser.add_argument('--num_workers', type=int, default=0)
    arg_parser.add_argument('--embed_size', type=int, default=128)
    arg_parser.add_argument('--hidden_size', type=int, default=512)
    arg_parser.add_argument('--tune', action='store_true', default=False)
    arg_parser.add_argument('--checkpoint_path', type=str)
    # add trainer args (gpus=x, precision=...)
    arg_parser = Trainer.add_argparse_args(arg_parser)
    args = arg_parser.parse_args()
    data = QuackTokenizedDataModule(args.data_dir, batch_size=args.batch_size, workers=args.num_workers)
    # Max time difference determined by data analysis.
    max_index = 132 + QuackConstants.VOCAB.value
    model = QuackAutoEncoder(num_embeddings=max_index, embed_size=args.embed_size, hidden_size=args.hidden_size, max_decode_length=data.get_width())
    if args.tune:
        trainer = Trainer.from_argparse_args(args, precision=16, auto_scale_batch_size=True)
        print('Ready for tuning...')
        trainer.tune(model, datamodule=data)
    else:
        trainer = Trainer.from_argparse_args(args, precision=16)
        print('Ready for training...')
        if args.checkpoint_path is None:
            trainer.fit(model, datamodule=data)
        else:
            trainer.fit(model, datamodule=data, ckpt_path=args.checkpoint_path)


if __name__ == '__main__':
    main()
