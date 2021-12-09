import h5py
import numpy as np
from bisect import bisect_left
import torch
from torch.utils.data import Dataset
from typing import Iterator, List, Tuple, Union
from cp_flatten import TokenizedQuackData, QuackConstants


class QuackIterableDataset(Dataset):
    """
    Iterates or selectively retrieves items from a collection of HDF5 files.  Metadata for the file is stored as
    HDF5 Attributes:
    length: The number of responses in the file.
    censored: The number of responses labeled 'censored' by existing Censored Planet process. Dataset must have been
        flattened as "Labeled"
    undetermined: The number of unlabeled responses.
    uncensored The number of responses labeled 'censored' by existing Censored Planet process. Dataset must have been
        flattened as "Labeled"

    Each response is stored in an HDF5 Group, named with the index number of the response, zero based.
    Metadata for the response is stored in HDF5 Attributes on the Group:
    domain: The domain under test
    ip: The IPv4 address for this test
    location: The country returned by MMDB for the IP address
    timestamp: A Unix timestamp for the time of the test
    censored: 1 if censored, -1 if uncensored, 0 as default (undetermined)

    Each response Group stores two HDF5 Datasets:
    'static_size': Data that is a fixed size.  See cp_flatten.CensoredPlanetFlatten.__process_row
    'variable_text' Text data that has been encoded (tokenized) using the XLMR pretrained model.
        See cp_flatten.CensoredPlanetFlatten.__process_row
    """

    def __init__(self, paths: List[str], tensors=False) -> None:
        """

        Parameters
        ----------
        paths: List[str]
            A list of paths as strings.
        """
        super().__init__()

        assert (
                paths is not None and isinstance(paths, list)
        ), "Must supply a set of file paths as a list"
        # Store the tensors flag. See __load_item().
        self.__as_tensors = tensors
        # Initialize parameters:
        self.__length = 0
        self.__censored = 0
        self.__undetermined = 0
        self.__uncensored = 0
        self.__max_width = 0
        self.__paths = paths
        self.__path_breakpoints = []
        self.__intermediate_lengths = []
        # Data indices are zero based.
        # Need to adjust total length by 1 to get correct index breakpoints.
        breakpoint_length = -1
        for path in paths:
            # Get metadata.
            with h5py.File(path, 'r') as storage:
                self.__length += storage.attrs['length']
                self.__censored += storage.attrs['censored']
                self.__undetermined += storage.attrs['undetermined']
                self.__uncensored += storage.attrs['uncensored']
                static_size = storage.attrs['static_size']
                variable_size = storage.attrs['max_text']
                width = static_size + variable_size
                if width > self.__max_width:
                    self.__max_width = width
                breakpoint_length += storage.attrs['length']
            self.__path_breakpoints.append(breakpoint_length)
            self.__intermediate_lengths.append(self.__length)

    def __iter__(self) -> Iterator[TokenizedQuackData]:
        """
        Iterates through all data points in the dataset.

        Returns
        -------

        """
        for index in range(self.__length):
            relative_index, file_path = self.__locate_item(index)
            with h5py.File(file_path, 'r') as storage:
                item = self.__load_item(relative_index, storage)
            yield item

    def __getitem__(self, index) -> TokenizedQuackData:
        """
        Implements a required method to access a single data point by index.
        """
        relative_index, file_path = self.__locate_item(index)
        with h5py.File(file_path, 'r') as storage:
            item = self.__load_item(relative_index, storage)
        return item

    def __len__(self) -> int:
        return self.__length

    def __load_item(self, index: int, storage: h5py.File) -> Union[TokenizedQuackData, torch.Tensor]:
        """
        Loads an item from an HDF5 file based on index value (group name).

        Parameters
        ----------
        index
        storage

        Returns
        -------

        """
        group_name = str(index)
        group = storage[group_name]
        if isinstance(group, h5py.Group):
            # Cast HDF5 datasets to numpy arrays, since Pytorch can create tensors directly from ndarray.
            dataset = group['static_size'] # type: h5py.Dataset
            static_source = np.zeros(dataset.shape, dataset.dtype)
            static_size = []
            dataset.read_direct(static_source)
            dataset = group['variable_text'] # type: h5py.Dataset
            variable_text = np.zeros(dataset.shape, dataset.dtype)
            dataset.read_direct(variable_text)
            # Create an "start marker" XLM-R uses 0, so will we.
            start = np.zeros(1, static_source.dtype)
            # Create an "end marker" XLM-R uses 2, so will we.
            end = np.full(1, 2, static_source.dtype)
            # Build metadata dictionary.
            meta = {}
            for key, value in group.attrs.items():
                meta[key] = value
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
                metadata=meta,
                static_size=static_source,
                variable_text=variable_text
            )

    def __locate_item(self, index: int) -> Tuple[int, str]:
        """
        Translates a global index value into a file relative index and provides a path to the file.

        Parameters
        ----------
        index: int
          The index in the range of items for all the files combined.

        Returns
        -------
        index: int
          The correct internal index for the item in the referenced file.
        path: str
          A file path to the file containing the item.
        """
        path_index = bisect_left(self.__path_breakpoints, index)
        path = self.__paths[path_index]
        if path_index > 0:
            # Subtract the lengths of the previous files.
            index = index - self.__intermediate_lengths[path_index - 1]
        return index, path

    def censored(self) -> int:
        return self.__censored

    def undetermined(self) -> int:
        return self.__undetermined

    def uncensored(self) -> int:
        return self.__uncensored

    def data_width(self) -> int:
        return self.__max_width
