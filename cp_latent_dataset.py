"""
Defines a class and data structures for quack data en tensors.
"""
import numpy as np
import torch
from pathlib import Path
import pickle
from torch.utils.data import Dataset
from typing import Iterator, Union, TypedDict


class QuackLatentData(TypedDict):
    """
    Encoded quack data and associated metadata.
    """
    metadata: dict
    encoded: np.ndarray


class QuackLatentDataset(Dataset):
    """
    Iterates or selectively retrieves items from a collection of python pickle files which contain QuackLatentData

    Metadata stored in metadata.pyc:

    length
        The number of responses in the file.
    censored
        The number of responses labeled 'censored' by existing Censored Planet process. Dataset must have been
        flattened as "Labeled"
    undetermined
        The number of unlabeled responses.
    uncensored
        The number of responses labeled 'censored' by existing Censored Planet process. Dataset must have been
        flattened as "Labeled"

    Each response is stored in a single .pyc file, named with the index number of the response, zero based.
    Metadata for the response is stored in the `metadata` key of the QuackImageData typed dictionary:

    domain
        The domain under test
    ip
        The IPv4 address for this test
    location
        The country returned by MMDB for the IP address
    timestamp
        A Unix timestamp for the time of the test
    censored
        1 if censored, -1 if uncensored, 0 as default (undetermined)

    Each QuackLatentData stores one numpy array:

    pixels
        A (224, 224) numpy array of pixel data

    See Also
    --------
    `cp_flatten.CensoredPlanetFlatten.__process_row`
    cp_image_reprocessor.py
    """

    def __init__(self, path: str) -> None:
        """
        Constructs QuackImageDataset.

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
        self.__path = path
        with open(self.__path + '/metadata.pyc', 'rb') as retrieved_dict:
            metadata = pickle.load(retrieved_dict)
        self.__length = metadata['length']
        self.__censored = metadata['censored']
        self.__undetermined = metadata['undetermined']
        self.__uncensored = metadata['uncensored']

    def __iter__(self) -> Iterator[QuackLatentData]:
        """
        Iterates through all data points in the dataset.

        Returns
        -------
        QuackImageData
        """
        for index in range(self.__length):
            file_path = self.__locate_item(index)
            yield self.__load_item(file_path)

    def __getitem__(self, index) -> QuackLatentData:
        """
        Implements a required method to access a single data point by index.

        Parameters
        ----------
        index: int
          The index of the data item.

        Returns
        -------
        QuackImageData
        """
        file_path = self.__locate_item(index)
        return self.__load_item(file_path)

    def __len__(self) -> int:
        return self.__length

    def __load_item(self, item_path: Path) -> QuackLatentData:
        """
        Loads an item from a pickle file.

        Parameters
        ----------
        item_path: Path
            A path object pointing to the item's storage.

        Returns
        -------
        QuackLatentData
        """
        with item_path.open(mode='rb') as storage:
            item = pickle.load(storage)
        return QuackLatentData(
            metadata=item['metadata'],
            encoded=item['encoded']
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
        return Path(f'{self.__path}/{segment_1}/{segment_2}/{index}.pyc')

    def censored(self) -> int:
        """
        Getter for the value of self.__censored.

        Returns
        -------
        int
        """
        return self.__censored

    def undetermined(self) -> int:
        """
        Getter for the value of self.__undetermined.

        Returns
        -------
        int
        """
        return self.__undetermined

    def uncensored(self) -> int:
        """
        Getter for the value of self.__uncensored.

        Returns
        -------
        int
        """
        return self.__uncensored
