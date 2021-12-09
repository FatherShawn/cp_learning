from cp_flatten import CensoredPlanetFlatten, TokenizedQuackData
import h5py
from datetime import datetime
from argparse import ArgumentParser
import numpy as np


class FreqIter:

    def __init__(self, source: dict) -> None:
        self.__frequency_dict = source

    def __iter__(self) -> int:
        for key, value in self.__frequency_dict.items():
            for instance in range(value):
                yield key


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
    dataset = CensoredPlanetFlatten(urls, args.vocab_path, True, True, True)
    count = 0
    variable_census = {}
    stats = {
        'censored': 0,
        'undetermined': 0,
        'uncensored': 0,
        'length': 0,
        'static_size': 0,
        'min_text': 0,
        'q1_text': 0,
        'median_text': 0,
        'q3_text': 0,
        'max_text': 0
    }
    with h5py.File(args.storage_path, 'w') as storage:
        for item in dataset:
            # Validate:
            meta = item['metadata']
            verify_returned_item(item)
            # Create response group
            index = str(count)
            # Store:
            response = storage.create_group(index)
            for key, value in meta.items():
                response.attrs[key] = value
            response.create_dataset('static_size', data=item['static_size'])
            response.create_dataset('variable_text', data=item['variable_text'])
            # Count:
            if meta['censored'] == 1:
                stats['censored'] += 1
            elif meta['censored'] == 0:
                stats['undetermined'] += 1
            elif meta['censored'] == -1:
                stats['uncensored'] += 1
            variable_size = item['variable_text'].size
            try:
                variable_census[variable_size] += 1
            except KeyError:
                variable_census[variable_size] = 1
            count += 1
            if count % 100000 == 0:
                # Really only need to store this once, but this is better than another conditional.
                stats['static_size'] = item['static_size'].size
                with open(args.log_path, 'a') as log:
                    item_date = datetime.fromtimestamp(meta['timestamp']).date().isoformat()
                    log.write(f'Processed {count:,} items. Last item processed was from {item_date}\n')
        census_expanded = FreqIter(variable_census)
        census = np.fromiter(census_expanded, int, count)
        stats['length'] = count
        stats['min_text'] = np.min(census)
        stats['q1_text'] = np.quantile(census, 0.25)
        stats['median_text'] = np.quantile(census, 0.5)
        stats['q3_text'] = np.quantile(census, 0.75)
        stats['max_text'] = np.max(census)
        for key in stats.keys():
            storage.attrs[key] = stats[key]

    # Report
    with open(args.log_path, 'a') as log:
        log.write(f'{count} items in the dataset with the following distribution:\n')
        for key, value in stats.items():
            log.write(f'{key}: {value}\n')


if __name__ == '__main__':
    main()
