import sys
import json
import tarfile
import re
from io import BufferedReader
from typing import Callable, Tuple, Iterator
from collections import deque
from torch.utils.data import IterableDataset
import webdataset as wds

def quack_file_expander(data: IterableDataset, handler: Callable = wds.tariterators.reraise_exception)-> Iterator[Tuple[str, object]]:
    """
    An adaptation of webdataset.tariterators.tar_file_expander that calls our quack_tar_file_iterator
    instead.

    Parameters
    ----------
    data : IterableDataset
        An dataset passed from url_opener.
    handler: Callable
        The exception handler.

    Yields
    -------
    tuple
        A tuple with 2 members:
        - str: filename
        - tarfile._Stream: An internal class used by the tarfile module
    """
    for source in data:
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in quack_tar_file_iterator(source["stream"]):
                assert isinstance(sample, tuple) and len(sample) == 2
                yield sample
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


def quack_tar_file_iterator(fileobj: BufferedReader, skip_meta: str = r".*\.txt$", handler: Callable = wds.tariterators.reraise_exception)-> Iterator[Tuple[str, object]]:
    """
    An adaptation of webdataset.tariterators.tar_file_expander that returns a stream to files in which we are interested
    rather than reading the entire file.

    Parameters
    ----------
    fileobj : BufferedReader
        A tarfile stream to the tar file.
    skip_meta : str
        A regular expression for file names to skip.
    handler: Callable
        The exception handler.

    Yields
    -------
    tuple
        A tuple with 2 members:
        - str: filename
        - tarfile._Stream: An internal class used by the tarfile module

    """
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    for tarinfo in stream:
        try:
            if not tarinfo.isreg():
                continue
            file_name = tarinfo.name
            if file_name is None:
                continue
            if (
                "/" not in file_name
                and file_name.startswith(wds.tariterators.meta_prefix)
                and file_name.endswith(wds.tariterators.meta_suffix)
            ):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, file_name):
                continue
            yield file_name, stream.extractfile(tarinfo)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
    del stream

def flatten_quack(source: IterableDataset, handler: Callable = wds.tariterators.reraise_exception ) -> dict:
    """
    Iterates through the data provided by the stream within source. Reads one line at a time, extracting the 'Results',
    which is a list of dictionaries with the result of each attempt for the current record.  The remainder of the record
    is treated as metadata and the record is flattened by merging each result with this metadata, which is pushed onto a
    stack.

    At each iteration the stack is popped and the result returned.  When the stack is empty, the next line is read and
    the stack is repopulated.

    Parameters
    ----------
    source : tuple
         A tuple with 2 members:
        - str: filename
        - tarfile._Stream: An internal class used by the tarfile module

    handler: Callable
        The exception handler.

    Yields
    -------
    dict
        Currently all values are strings.  Types shown are inferred. Keys used are:
            - Server: (str) An IPv4 address
            - Keyword: (str) An FQDN
            - Retries: (int) A count of additional requests
            - Blocked: (bool)
            - FailSanity: (bool)
            - StatefulBlock: (bool)
            - Sent: (str) The request
            - Received: (str) [optional] The response received.  Present if `Success` is False.
            - Success: (bool)
            - Error: (bool)
            - StartTime: A time stamp in the format: Complete date plus hours, minutes, seconds and a decimal fraction
              of a second YYYY-MM-DDThh:mm:ss.sTZD (eg 2021-02-28T06:51:06.173912969-05:00)
            - EndTime: A time stamp in the format: Complete date plus hours, minutes, seconds and a decimal fraction
              of a second YYYY-MM-DDThh:mm:ss.sTZD (eg 2021-02-28T06:51:06.173912969-05:00)

    """
    data = deque()
    for file_name, connection in source:
        iterate_lines = True
        while iterate_lines:
            try:
                if len(data) == 0:
                    # Repopulate the data list.
                    line = connection.readline()
                    if line == b'':
                        raise StopIteration
                    datum = json.loads(line.decode('UTF-8'))
                    try:
                        results = datum.pop('Results')
                    except KeyError:
                        results = {}
                    for result in results:
                        data.append(datum | result)
                yield data.pop()
            except StopIteration:
                iterate_lines = False
            except Exception as exn:
                handler(exn)


def main(argv):
    filename = argv[1]
    url = 'file:{path}'.format(path=filename)
    dataset = wds.ShardList(url)
    dataset = wds.Processor(dataset, wds.url_opener)
    dataset = wds.Processor(dataset, quack_file_expander)
    dataset = wds.Processor(dataset, flatten_quack)

    for raw_result in dataset:
        ######
        # The raw_result returns as a dictionary with this structure - vectorization notes for each field are given:
        # 'Server': Encode to 4 element tensor,
        # 'Keyword': TBD, label or metadata? 'www.worldcoalition.org',
        # 'Retries': Encode to single element tensor,
        # 'Blocked': Encode to single element tensor,
        # 'FailSanity': False,
        # 'StatefulBlock': Encode to single element tensor,
        # 'Sent': Encode as tokens using XLM-R,
        # 'Received': Encode as tokens using XLM-R. Not present in the data if Success is true - use `Sent` in that case?
        # 'Success': Encode to single element tensor,
        # 'Error': Encode as tokens using XLM-R,
        # 'StartTime': Encode to single element tensor as unix timestamp,
        # 'EndTime': Encode to single element tensor as unix timestamp
        #
        # Rearrange the order to put the variable length items at the end?  What's the proper way to prepare variable
        # length data for a Variational Auto Encoder?
        #
        # Conclusion: Pytorch is explicitly designed for this issue:
        # [dynamic computational graph](https://medium.com/intuitionmachine/pytorch-dynamic-computational-graphs-and-modular-deep-learning-7e7f89f18d1)
        # So place the variable length data at the end of the input and process one input at a time.  It appears that
        # only when processing multiple inputs as a batch does Pytorch need input data to be of a consistent size.
        ######
        pass

if __name__ == '__main__':
    main(sys.argv)