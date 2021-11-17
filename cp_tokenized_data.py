import torch as pt
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from cp_dataset import QuackIterableDataset
from typing import Optional


def pad_right(batch: list[pt.Tensor]) -> pt.Tensor:
    '''
    Receives a list of Tensors with B elements.  Calculates the widest tensor, sorting the longest length T. Pads all
    narrower tensors to T with zeros.  Returns a (B x T) shaped tensor.

    Parameters
    ----------
    batch: list[pt.Tensor]
        A list of tensors in the batch.

    Returns
    -------
    pt.Tensor
    '''
    lengths = np.fromiter((item.size(0) for item in batch), int)
    max_length = np.max(lengths)
    batch_padded = [F.pad(item, (0, max_length - item.size(0)), value=0) for item in batch]
    return pt.stack(batch_padded)


class QuackTokenizedDataModule(pl.LightningDataModule):
    def __init__(self, train_paths: list, validation_paths: list, batch_size: int = 64, train_transforms=None,
                 val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.__train_paths = train_paths
        self.__val_paths = validation_paths
        self.__batch_size = batch_size
        dataset = QuackIterableDataset(self.__train_paths, tensors=True)
        width = dataset.data_width()
        # Reserve 20% of the data as test data.
        test_reserve = int(len(dataset)*0.2)
        self.__train_data, self.__test_data = random_split(dataset, [len(dataset) - test_reserve, test_reserve])
        self.__val_data = QuackIterableDataset(self.__val_paths, tensors=True)
        if self.__val_data.data_width() > width:
            width = self.__val_data.data_width()
        self._dims = (width,)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.__train_data, batch_size=self.__batch_size, collate_fn=pad_right, num_workers=4)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.__test_data, batch_size=self.__batch_size, collate_fn=pad_right, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.__val_data, batch_size=self.__batch_size, collate_fn=pad_right, num_workers=4)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

