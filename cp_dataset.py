import json
import numpy
import os
import shutil
import torch
import tarfile
from io import BytesIO
from pathlib import Path
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
from typing import Iterator
from cp_flatten import TokenizedQuackData
from webdataset import ShardList, Shorthands

from cp_processor import STORAGE_PATH


class QuackShards(IterableDataset, Shorthands):
    """
    """

    def __init__(self, url: str) -> None:
        super().__init__()

        assert (
                url is not None
        ), "Must supply a url as a string to the dataset directory"
        self.__url = url
        # Get metadata.
        with open(f'{url}/cp_dataset.json', 'r') as dataset_metadata:
            metadata = json.load(dataset_metadata)
            self.__shard_size = metadata['shard_size']
            self.__length = metadata['length']
        # Get shard count.
        shards = metadata['shards']
        urls = []
        for index in range(shards + 1):
             urls.append(f'{self.__url}/{index // 1000}/{index}')
        self.__shards = ShardList(urls)
        # Set the length.


    def __iter__(self) -> Iterator[TokenizedQuackData]:
        for shard in self.__shards:
            for response in Path(shard['url']).iterdir():
                with open(response / 'metadata.json', 'r') as response_metadata:
                    metadata = json.load(response_metadata)
                static_size = numpy.load(response / 'static_size.npy')
                variable_text = numpy.load(response / 'variable_text.npy')
                yield TokenizedQuackData(
                    metadata=metadata,
                    static_size=static_size,
                    variable_text=variable_text
                )

    def __getitem__(self, index) -> T_co:
        """
        Required by the parent of IterableDataset but not useful in this context, and not implemented by any of the
        Webdataset implementations of IterableDataset.
        """
        shard = index // self.__shard_size
        response_id = index % self.__shard_size - 1
        url = f'{self.__url}/quack-{shard}.tar'
        with tarfile.open(url, mode="r") as tarball:
            target = tarball.getmember(f'response-{response_id}.metadata.json')
            target_data = tarball.extractfile(target).read()
        stream = BytesIO(target_data)
        return torch.load(stream)

    def __len__(self) -> int:
        return self.__length