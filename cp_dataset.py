import json
import torch
import tarfile
from io import BytesIO
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
from typing import Iterator, Tuple
from cp_flatten import MetaTensor
from webdataset import ShardList, Shorthands, tariterators, url_opener, autodecode

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
            self.__shard_size = metadata['shards']
            self.__length = metadata['length']
        # Calculate shards.
        shards = self.__length // self.__shard_size
        urls = f'{self.__url}/quack-{{0..{shards}}}.tar'
        self.__shards = ShardList(urls)
        # Set the length.


    def __iter__(self) -> Iterator[MetaTensor]:
        for shard in url_opener(self.__shards):
            current_url = shard['url']
            print(f'Processing {current_url}:')
            # Use a small stack to build a set of 3 files.
            # Each processed item is stored as 3 files in the tarball
            response_set = []
            for filename, data in tariterators.tar_file_expander([shard]):
                if not 'response-' in filename:
                    continue
                stream = BytesIO(data)
                yield torch.load(stream)

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
