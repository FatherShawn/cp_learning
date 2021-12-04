from cp_flatten import QuackConstants
from cp_tokenized_data import QuackTokenizedDataModule
from autoencoder import QuackAutoEncoder
from pytorch_lightning import Trainer


def main() -> None:
    # A list of paths to HDF5 files.
    data_paths = [
        '/data/unlabeled/2021-10-13.hdf5',
        '/data/labeled/2021-08-04-labeled.hdf5',
        '/data/labeled/2021-08-08-labeled.hdf5',
        '/data/labeled/2021-08-11-labeled.hdf5',
        '/data/labeled/2021-08-16-labeled.hdf5',
        '/data/labeled/2021-08-25-labeled.hdf5'
    ]
    data = QuackTokenizedDataModule(data_paths, batch_size=64)
    # Max time difference determined by data analysis.
    max_index = 131300 + QuackConstants.VOCAB.value
    model = QuackAutoEncoder(num_embeddings=max_index, embed_size=32, hidden_size=128, max_decode_length=data.get_width())
    trainer = Trainer(gpus=1, precision=16, limit_train_batches=0.1)
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    main()
