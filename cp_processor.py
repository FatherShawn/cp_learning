from cp_flatten import CensoredPlanetFlatten, TokenizedQuackData
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import pickle


class FreqIter:

    def __init__(self, source: dict) -> None:
        self.__frequency_dict = source

    def __iter__(self) -> int:
        for key, value in self.__frequency_dict.items():
            for instance in range(value):
                yield key


def filepath(index: int, dir_only=False) -> str:
    rank_five = index // 100000
    remainder = index - (rank_five * 100000)
    rank_three_four = remainder // 1000
    if dir_only:
        return f'/{rank_five}/{rank_three_four}'
    return f'/{rank_five}/{rank_three_four}/{index}.pyc'


def verify_returned_item(item: TokenizedQuackData) -> None:
    meta = item['metadata']
    assert (isinstance(item, dict)), 'Item from the dataset is not a dictionary.'
    assert ('metadata' in item), 'Key "metadata" not found in item from the dataset.'
    assert ('static_size' in item), 'Key "static_size" not found in item from the dataset.'
    assert ('variable_text' in item), 'Key "variable_text" not found in item from the dataset.'
    assert isinstance(meta, dict)
    assert len(meta) == 5
    assert (isinstance(item['static_size'], np.ndarray)), 'static_size is not a numpy array'
    assert (isinstance(item['variable_text'], np.ndarray)), 'variable_text is not a numpy array'
    assert (meta['censored'] in (1, 0, -1)), 'censored value is out of bounds'


def main() -> None:
    """
    Create a list of file paths or urls to process.  The webdataset library expects a list,
    so we place only one item in the list.
    """
    # Add args to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--source_path', type=str, required=True)
    arg_parser.add_argument('--storage_path', type=str, required=True)
    arg_parser.add_argument('--log_path', type=str, required=True)
    arg_parser.add_argument('--vocab_path', type=str, required=True)
    args = arg_parser.parse_args()

    urls = [
        args.source_path
    ]
    dataset = CensoredPlanetFlatten(urls, args.vocab_path, True, False, True)
    count = 0
    variable_census = {}

    try:
        with open(args.storage_path + '/metadata.pyc', 'rb') as retrieved_dict:
            metadata = pickle.load(retrieved_dict)
            count = metadata['length']
    except OSError:
        count = 0
        metadata = {
            'censored': 0,
            'undetermined': 0,
            'uncensored': 0,
            'length': 0,
            'max_width': 0
        }

    for item in dataset:
        # Validate:
        meta = item['metadata']
        verify_returned_item(item)
        # Create response group
        index = str(count)
        # Ensure storage is ready.
        storage_path = Path(args.storage_path + filepath(count, dir_only=True))
        storage_path.mkdir(parents=True, exist_ok=True)
        item_storage = Path(args.storage_path + filepath(count))
        # Count:
        if meta['censored'] == 1:
            metadata['censored'] += 1
        elif meta['censored'] == 0:
            metadata['undetermined'] += 1
        elif meta['censored'] == -1:
            metadata['uncensored'] += 1
        width = item['static_size'].size + item['variable_text'].size
        if width > metadata['max_width']:
            metadata['max_width'] = width
        # Store as verified:
        verified = False
        with item_storage.open(mode='wb') as target:
            pickle.dump(item, target)
        try:
            with item_storage.open(mode='rb') as check:
                check = pickle.load(check)
            verify_returned_item(check)
        except Exception:
            # Don't sweat a single failure among 100s of thousands of items.
            continue
        count += 1
        if count % 100000 == 0:
            with open(args.log_path, 'a') as log:
                item_date = datetime.fromtimestamp(meta['timestamp']).date().isoformat()
                log.write(f'Processed {count:,} items. Last item processed was from {item_date}\n')
    metadata['length'] = count
    with open(args.storage_path + '/metadata.pyc', 'wb') as stored_dict:
        pickle.dump(metadata, stored_dict)
    print(f'{count} items processed into pickled dictionaries')
    for key, value in metadata.items():
        print(f'{key}: {value}')

    # Report
    with open(args.log_path, 'a') as log:
        log.write(f'{count} items processed into pickled dictionaries with the following metadata:\n')
        for key, value in metadata.items():
            log.write(f'{key}: {value}\n')


if __name__ == '__main__':
    main()
