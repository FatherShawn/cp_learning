from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
from typing import Iterator, Union
from cp_flatten import MetaTensor
from webdataset import ShardList, Shorthands, tariterators, url_opener, autodecode

class QuackShards(IterableDataset, Shorthands):
    """
    """

    def __init__(self, urls: Union[str, list[str]]) -> None:
        super().__init__()

        assert (
                urls is not None
        ), "Must supply a url as a string or list of strings"

        self.__shards = ShardList(urls)

    def __iter__(self) -> Iterator[MetaTensor]:
        for shard in url_opener(self.__shards):
            current_url = shard['url']
            print(f'Processing {current_url}:')
            for filename, data in tariterators.tar_file_expander([shard]):
                yield autodecode.basichandlers(filename, data)

    def __getitem__(self, index) -> T_co:
        """
        Required by the parent of IterableDataset but not useful in this context, and not implemented by any of the
        Webdataset implementations of IterableDataset.
        """
        pass
