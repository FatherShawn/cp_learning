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
    arg_parser.add_argument('--limit_training', type=float, default=1.0)
    args = arg_parser.parse_args()
    # A list of paths to HDF5 files.
    data_paths = [
        args.data_dir + '2021-08-04-labeled.hdf5',
        args.data_dir + '2021-08-26-labeled.hdf5',
        args.data_dir + '2021-08-25-labeled.hdf5',
        args.data_dir + '2021-08-08-labeled.hdf5',
        args.data_dir + '2021-08-16-labeled.hdf5',
        args.data_dir + '2021-10-13-unlabled.hdf5'
    ]
    data = QuackTokenizedDataModule(data_paths, batch_size=args.batch_size, workers=args.num_workers)
    # Max time difference determined by data analysis.
    max_index = 131300 + QuackConstants.VOCAB.value
    model = QuackAutoEncoder(num_embeddings=max_index, embed_size=args.embed_size, hidden_size=args.hidden_size, max_decode_length=data.get_width())
    trainer = Trainer(limit_train_batches=0.05)
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    main()
