import numpy as np
import torch
from pathlib import Path
import pickle
from torch.utils.data import Dataset
from typing import Iterator, Union, TypedDict


class QuackImageData(TypedDict):
    metadata: dict
    pixels: np.ndarray


class QuackImageDataset(Dataset):
    """
    Iterates or selectively retrieves items from a collection of python pickle files which contain QuackImageData
    length: The number of responses in the file.
    censored: The number of responses labeled 'censored' by existing Censored Planet process. Dataset must have been
        flattened as "Labeled"
    undetermined: The number of unlabeled responses.
    uncensored The number of responses labeled 'censored' by existing Censored Planet process. Dataset must have been
        flattened as "Labeled"

    Each response is stored in a single .pyc file, named with the index number of the response, zero based.
    Metadata for the response is stored in the metadata key of the TokenizedQuackData typed dictionary:
    domain: The domain under test
    ip: The IPv4 address for this test
    location: The country returned by MMDB for the IP address
    timestamp: A Unix timestamp for the time of the test
    censored: 1 if censored, -1 if uncensored, 0 as default (undetermined)

    Each QuackImageData stores one numpy arrays:
    'pixels': A (224, 224) numpy array of pixel data
        See cp_flatten.CensoredPlanetFlatten.__process_row
        See cp_image_reprocessor.py
    """

    def __init__(self, path: str) -> None:
        """

        Parameters
        ----------
        paths: str
            A path to the top level of the data directories.
        """
        super().__init__()

        assert (path is not None), "Must supply a file path"
        # Initialize parameters:
        self.__length = 0
        self.__censored = 0
        self.__undetermined = 0
        self.__uncensored = 0
        self.__max_width = 0
        self.__path = path
        with open(self.__path + '/metadata.pyc', 'rb') as retrieved_dict:
            metadata = pickle.load(retrieved_dict)
        self.__length = metadata['length']
        self.__censored = metadata['censored']
        self.__undetermined = metadata['undetermined']
        self.__uncensored = metadata['uncensored']

    def __iter__(self) -> Iterator[QuackImageData]:
        """
        Iterates through all data points in the dataset.

        Returns
        -------

        """
        for index in range(self.__length):
            file_path = self.__locate_item(index)
            yield self.__load_item(file_path)

    def __getitem__(self, index) -> QuackImageData:
        """
        Implements a required method to access a single data point by index.
        """
        file_path = self.__locate_item(index)
        return self.__load_item(file_path)

    def __len__(self) -> int:
        return self.__length

    def __load_item(self, item_path: Path) -> Union[QuackImageData, torch.Tensor]:
        """
        Loads an item from a pickle file.

        Parameters
        ----------
        item_path: Path
            A path object pointing to the item's storage.

        Returns
        -------

        """
        with item_path.open(mode='rb') as storage:
            item = pickle.load(storage)
        return QuackImageData(
            metadata=item['metadata'],
            pixels=item['pixels']
        )

    def __locate_item(self, index: int) -> Path:
        """
        Translates a global index value into a file path to the file or enclosing directory.

        Parameters
        ----------
        index: int
          The index in the range of items for all the files combined.

        Returns
        -------
        path: Path
          A path object to the file containing the item.
        """
        segment_1 = index // 100000
        remainder = index - (segment_1 * 100000)
        segment_2 = remainder // 1000
        stem = f'/{segment_1}/{segment_2}/{index}/{index}.pyc'
        return Path(f'{self.__path}/{stem}')

    def censored(self) -> int:
        return self.__censored

    def undetermined(self) -> int:
        return self.__undetermined

    def uncensored(self) -> int:
        return self.__uncensored

    def data_width(self) -> int:
        return self.__max_width
