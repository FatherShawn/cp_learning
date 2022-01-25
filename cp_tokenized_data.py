from typing import Tuple
import torch as pt
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from cp_dataset import QuackIterableDataset
from cp_flatten import QuackConstants
from typing import List


def pad_right(batch: List[dict]) -> pt.Tensor:
    '''
    Receives a list of Tensors with B elements.  Calculates the widest tensor, which is length T. Pads all
    narrower tensors to T with zeros.  Returns a (B x T) shaped tensor.

    Parameters
    ----------
    batch: List[pt.Tensor]
        A list of tensors in the batch.

    Returns
    -------
    pt.Tensor
    '''
    data = []
    for item in batch:
        data.append(pt.from_numpy(concatenate_data(item)))
    lengths = np.fromiter((item.size(0) for item in data), int)
    max_length = np.max(lengths)
    batch_padded = [F.pad(item, (0, max_length - item.size(0)), value=QuackConstants.XLMR_PAD.value) for item in data]
    return pt.stack(batch_padded)

def pad_right_with_meta(batch: List[dict]) -> Tuple[List[dict], pt.Tensor]:
    '''
    Receives a list of TokenizedQuackData with B elements.  Calculates the widest tensor, which is length T. Pads all
    narrower tensors to T with zeros.  Returns a (B x T) shaped tensor.

    Parameters
    ----------
    batch: List[TokenizedQuackData]
        A list of TokenizedQuackData (TypedDict) in the batch.

    Returns
    -------
    pt.Tensor
    '''
    data = []
    meta = []
    for item in batch:
        data.append(pt.from_numpy(concatenate_data(item)))
        meta.append(item['metadata'])
    lengths = np.fromiter((item.size(0) for item in data), int)
    max_length = np.max(lengths)
    batch_padded = [F.pad(item, (0, max_length - item.size(0)), value=QuackConstants.XLMR_PAD.value) for item in data]
    return meta, pt.stack(batch_padded)


def concatenate_data(item: dict) ->np.ndarray:
    static_source = item['static_size']  # type: np.ndarray
    static_size = []
    variable_text = item['variable_text']  # type: np.ndarray
    # Create an "start marker" XLM-R uses 0, so will we.
    start = np.zeros(1, static_source.dtype)
    # Create an "end marker" XLM-R uses 2, so will we.
    end = np.full(1, 2, static_source.dtype)
    # Build the sequence as a tensor, text first.
    # Time values at static_source index 8 & 9.
    time_values = {8, 9}
    for index in range(static_source.size):
        if index in time_values:
            continue
        # Shift value by vocabulary size to avoid value collisions.
        static_size.append(int(static_source[index] + QuackConstants.VOCAB.value))
    # Now deal with time by finding the difference in seconds.
    time_diff = round((static_source[9] - static_source[8]))
    static_size.append(time_diff + QuackConstants.VOCAB.value)
    return np.concatenate((variable_text, start, np.array(static_size), end), dtype=static_source.dtype).astype(
        np.int_)

class QuackTokenizedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, workers: int = 0, train_transforms=None,
                 val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.__batch_size = batch_size
        self.__workers = workers
        dataset = QuackIterableDataset(data_dir, tensors=True)
        self.__predict_data = QuackIterableDataset(data_dir)
        print(f'Source dataset ready with {len(dataset)} items.')
        self.__width = dataset.data_width()
        # Reserve 20% of the data as test data.
        test_reserve = round(len(dataset) * 0.2)
        # Reserve 10% of the data as validation data.
        val_reserve = round(len(dataset) * 0.1)
        self.__train_data, self.__test_data, self.__val_data = random_split(
            dataset, [len(dataset) - test_reserve - val_reserve, test_reserve, val_reserve]
        )
        print(f'Training dataset randomly split with {len(self.__train_data)} items.')
        print(f'Test dataset randomly split with {len(self.__test_data)} items.')
        print(f'Validation dataset randomly split with {len(self.__val_data)} items.')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.__train_data, batch_size=self.__batch_size, collate_fn=pad_right, shuffle=True, num_workers=self.__workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.__test_data, batch_size=self.__batch_size, collate_fn=pad_right, num_workers=self.__workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.__val_data, batch_size=self.__batch_size, collate_fn=pad_right, num_workers=self.__workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.__predict_data, batch_size=self.__batch_size, collate_fn=pad_right_with_meta, num_workers=self.__workers)

    def get_width(self):
        return self.__width