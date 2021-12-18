import numpy as np
import torch
from pathlib import Path
import pickle
from torch.utils.data import Dataset
from typing import Iterator, List, Tuple, Union
from cp_flatten import TokenizedQuackData, QuackConstants


class QuackIterableDataset(Dataset):
    """
    Iterates or selectively retrieves items from a collection of python pickle files which contain TokenizedQuackData
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

    Each TokenizedQuackData stores two numpy arrays:
    'static_size': Data that is a fixed size.  See cp_flatten.CensoredPlanetFlatten.__process_row
    'variable_text' Text data that has been encoded (tokenized) using the XLMR pretrained model.
        See cp_flatten.CensoredPlanetFlatten.__process_row
    """

    def __init__(self, path: str, tensors=False) -> None:
        """

        Parameters
        ----------
        paths: str
            A path to the top level of the data directories.
        """
        super().__init__()

        assert (path is not None), "Must supply a file path"
        # Store the tensors flag. See __load_item().
        self.__as_tensors = tensors
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
        self.__max_width = metadata['max_width']

    def __iter__(self) -> Iterator[TokenizedQuackData]:
        """
        Iterates through all data points in the dataset.

        Returns
        -------

        """
        for index in range(self.__length):
            file_path = self.__locate_item(index)
            yield self.__load_item(file_path)

    def __getitem__(self, index) -> TokenizedQuackData:
        """
        Implements a required method to access a single data point by index.
        """
        file_path = self.__locate_item(index)
        return self.__load_item(file_path)

    def __len__(self) -> int:
        return self.__length

    def __load_item(self, item_path: Path) -> Union[TokenizedQuackData, torch.Tensor]:
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
        static_source = item['static_size']  # type: np.ndarray
        static_size = []
        variable_text = item['variable_text']  # type: np.ndarray
        # Create an "start marker" XLM-R uses 0, so will we.
        start = np.zeros(1, static_source.dtype)
        # Create an "end marker" XLM-R uses 2, so will we.
        end = np.full(1, 2, static_source.dtype)
        # Build the sequence as a tensor, text first.
        if self.__as_tensors:
            # Time values at static_source index 8 & 9.
            time_values = {8, 9}
            for index in range(static_source.size):
                if index in time_values:
                   continue
                # Shift value by vocabulary size to avoid value collisions.
                static_size.append(int(static_source[index] + QuackConstants.VOCAB.value))
            # Now deal with time by finding the difference in milliseconds.
            time_diff = round((static_source[9] - static_source[8]) * 1000)
            static_size.append(time_diff + QuackConstants.VOCAB.value)
            row = np.concatenate((variable_text, start, np.array(static_size), end), dtype=static_source.dtype).astype(np.int_)
            return torch.from_numpy(row)
        return TokenizedQuackData(
            metadata=item['metadata'],
            static_size=static_source,
            variable_text=variable_text
        )

    def __locate_item(self, index: int, dir_only: bool = False) -> Path:
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
        if dir_only:
            return Path(f'/{segment_1}/{segment_2}')
        return Path(f'{self.__path}/{segment_1}/{segment_2}/{index}.pyc')

    def censored(self) -> int:
        return self.__censored

    def undetermined(self) -> int:
        return self.__undetermined

    def uncensored(self) -> int:
        return self.__uncensored

    def data_width(self) -> int:
        return self.__max_width
