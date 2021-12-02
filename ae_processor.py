from cp_flatten import QuackConstants
from cp_tokenized_data import QuackTokenizedDataModule
from autoencoder import QuackAutoEncoder
from pytorch_lightning import Trainer


def main() -> None:
    # A list of paths to HDF5 files.
    train_paths = [
        '/data/unlabeled/2021-09-30.hdf5',
        '/data/unlabeled/2021-10-11.hdf5',
        '/data/unlabeled/2021-10-13.hdf5',
        '/data/unlabeled/2021-10-14.hdf5'
    ]
    validation_paths = [
        '/data/labeled/2021-08-04-labeled.hdf5',
        '/data/labeled/2021-08-08-labeled.hdf5',
        '/data/labeled/2021-08-11-labeled.hdf5',
        '/data/labeled/2021-08-16-labeled.hdf5',
        '/data/labeled/2021-08-25-labeled.hdf5'
    ]
    data = QuackTokenizedDataModule(train_paths, validation_paths, batch_size=2)
    # Max time difference determined by data analysis.
    max_index = 131300 + QuackConstants.XLMR_VOCAB.value
    model = QuackAutoEncoder(num_embeddings=max_index, embed_size=128, hidden_size=512, max_decode_length=data.get_width())
    trainer = Trainer(gpus=1, precision=16)
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    main()
