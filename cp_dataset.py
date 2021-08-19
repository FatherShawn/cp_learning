import h5py
import json
import numpy as np
import torch
import tarfile
from io import BytesIO
from pathlib import Path
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
from typing import Iterator
from cp_flatten import TokenizedQuackData

from cp_processor import STORAGE_PATH


class QuackShards(IterableDataset):
    """
    """

    def __init__(self, path: str) -> None:
        super().__init__()

        assert (
                path is not None
        ), "Must supply a file path as a string to the dataset directory"
        self.__path = path
        # Get metadata.
        with h5py.File(self.__path, 'r') as storage:
            self.__length = storage.attrs['length']
            self.__censored = storage.attrs['censored']
            self.__undetermined = storage.attrs['undetermined']
            self.__uncensored = storage.attrs['uncensored']

    def __iter__(self) -> Iterator[TokenizedQuackData]:
        with h5py.File(self.__path, 'r') as storage:
            for index in range(self.__length):
                yield self.__load_item(index, storage)

    def __getitem__(self, index) -> TokenizedQuackData:
        """

        """
        with h5py.File(self.__path, 'r') as storage:
            item = self.__load_item(index, storage)
        return item

    def __len__(self) -> int:
        return self.__length

    def __load_item(self, index: int, storage: h5py.File) -> TokenizedQuackData:
        group_name = str(index)
        group = storage[group_name]
        if isinstance(group, h5py.Group):
            meta = {}
            for key, value in group.attrs.items():
                meta[key] = value
            return TokenizedQuackData(
                metadata=meta,
                static_size=np.array(group['static_size']),
                variable_text=np.array(group['variable_text'])
            )

    def censored(self) -> int:
        return self.__censored

    def undetermined(self) -> int:
        return self.__undetermined

    def uncensored(self) -> int:
        return self.__uncensored