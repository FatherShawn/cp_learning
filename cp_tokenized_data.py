import torch as pt
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from cp_dataset import QuackIterableDataset
from cp_flatten import QuackConstants
from typing import Optional


def pad_right(batch: list[pt.Tensor]) -> pt.Tensor:
    '''
    Receives a list of Tensors with B elements.  Calculates the widest tensor, which is length T. Pads all
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
    batch_padded = [F.pad(item, (0, max_length - item.size(0)), value=QuackConstants.XLMR_PAD.value) for item in batch]
    return pt.stack(batch_padded)


class QuackTokenizedDataModule(pl.LightningDataModule):
    def __init__(self, data_paths: list, batch_size: int = 64, workers: int = 0, train_transforms=None,
                 val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.__batch_size = batch_size
        self.__workers = workers
        dataset = QuackIterableDataset(data_paths, tensors=True)
        self.__width = dataset.data_width()
        # Reserve 20% of the data as test data.
        test_reserve = round(len(dataset) * 0.2)
        # Reserve 10% of the data as validation data.
        val_reserve = round(len(dataset) * 0.1)
        self.__train_data, self.__test_data, self.__val_data = random_split(
            dataset, [len(dataset) - test_reserve - val_reserve, test_reserve, val_reserve]
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.__train_data, batch_size=self.__batch_size, collate_fn=pad_right, shuffle=True, num_workers=self.__workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.__test_data, batch_size=self.__batch_size, collate_fn=pad_right, num_workers=self.__workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.__val_data, batch_size=self.__batch_size, collate_fn=pad_right, num_workers=self.__workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def get_width(self):
        return self.__width
