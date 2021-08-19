from cp_flatten import CensoredPlanetFlatten, TokenizedQuackData
import h5py
import json
import numpy
import os

SHARD_SIZE = 1000
STORAGE_PATH = '/data/quack-07.22.01-08.11.01.hdf5'
MAX_VERIFICATION_ATTEMPTS = 10

def verify_returned_item(item: TokenizedQuackData) -> None:
    meta = item['metadata']
    assert (isinstance(item, dict)), 'Item from the dataset is not a dictionary.'
    assert ('metadata' in item), 'Key "metadata" not found in item from the dataset.'
    assert ('static_size' in item), 'Key "static_size" not found in item from the dataset.'
    assert ('variable_text' in item), 'Key "variable_text" not found in item from the dataset.'
    assert isinstance(meta, dict)
    assert len(meta) == 5
    assert (isinstance(item['static_size'], numpy.ndarray)), 'static_size is not a numpy array'
    assert (isinstance(item['variable_text'], numpy.ndarray)), 'variable_text is not a numpy array'
    assert (meta['censored'] in (1, 0, -1)), 'censored value is out of bounds'


def main() -> None:
    urls = [
        '/data/quack/CP_Quack-echo-2021-08-11-01-01-01.tar.gz',
        '/data/quack/CP_Quack-echo-2021-08-09-01-01-01.tar.gz',
        '/data/quack/CP_Quack-echo-2021-08-04-01-01-01.tar.gz',
        '/data/quack/CP_Quack-echo-2021-08-02-01-01-01.tar.gz',
        '/data/quack/CP_Quack-echo-2021-08-08-01-01-01.tar.gz',
        '/data/quack/CP_Quack-echo-2021-07-29-01-01-01.tar.gz',
        '/data/quack/CP_Quack-echo-2021-07-28-01-01-01.tar.gz',
        '/data/quack/CP_Quack-echo-2021-07-26-01-01-01.tar.gz',
        '/data/quack/CP_Quack-echo-2021-07-22-01-01-01.tar.gz'
    ]
    dataset = CensoredPlanetFlatten(urls, True, True)
    count = 0
    stats = {
        'censored': 0,
        'undetermined': 0,
        'uncensored': 0
    }
    with h5py.File(STORAGE_PATH, 'w') as storage:
        for item in dataset:
            # Validate:
            meta = item['metadata']
            verify_returned_item(item)
            # Create response group
            index = str(count)
            response = storage.create_group(index)
            # Store:
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
            count += 1
        storage.attrs['length'] = count
        storage.attrs['censored'] = stats['censored']
        storage.attrs['undetermined'] = stats['undetermined']
        storage.attrs['uncensored'] = stats['uncensored']

    # Report
    print(f'{count} items in the dataset with the following distribution:')
    for key, value in stats.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    main()