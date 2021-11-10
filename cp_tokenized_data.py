import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import PackedSequence, pad_sequence, pack_padded_sequence
from cp_dataset import QuackIterableDataset
from typing import Optional


def pad_and_pack(batch: list[Tensor]) -> PackedSequence:
    lengths = [item.size(0) for item in batch]
    batch_padded = pad_sequence(batch, batch_first=True)
    sequence = pack_padded_sequence(batch_padded, lengths=lengths, batch_first=True, enforce_sorted=False)
    return sequence


class QuackTokenizedDataModule(pl.LightningDataModule):
    def __init__(self, train_paths: list, test_paths: list, batch_size: int = 64, train_transforms=None,
                 val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.__train_paths = train_paths
        self.__test_paths = test_paths
        self.__batch_size = batch_size
        self.__train_data = QuackIterableDataset(self.__train_paths, tensors=True)
        width = self.__train_data.data_width()
        self.__test_data = QuackIterableDataset(self.__test_paths, tensors=True)
        if self.__test_data.data_width() > width:
            width = self.__test_data.data_width()
        self._dims = (width,)



    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.__train_data, batch_size=self.__batch_size, collate_fn=pad_and_pack)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.__test_data, batch_size=self.__batch_size, collate_fn=pad_and_pack)
