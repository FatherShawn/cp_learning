from cp_flatten import CensoredPlanetFlatten, TokenizedQuackData
import json
import numpy
import os

SHARD_SIZE = 1000
STORAGE_PATH = ''
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

    ]
    dataset = CensoredPlanetFlatten(urls, True, True)
    count = 0
    shard = 0
    stats = {
        'censored': 0,
        'undetermined': 0,
        'uncensored': 0
    }
    for item in dataset:
        # Validate:
        meta = item['metadata']
        verify_returned_item(item)
        # Store:
        path = f'{STORAGE_PATH}/{shard // 1000}/{shard}/{count % SHARD_SIZE}'
        os.makedirs(path, 0o755, True)
        with open(f'{path}/metadata.json', 'w') as response_metadata:
            response_metadata.write(json.dumps(meta))
        numpy.savez(f'{path}/data', static_size=item['static_size'], variable_text=item['variable_text'])
        # Count:
        if meta['censored'] == 1:
            stats['censored'] += 1
        elif meta['censored'] == 0:
            stats['undetermined'] += 1
        elif meta['censored'] == -1:
            stats['uncensored'] += 1
        count += 1
        # Check.
        if count % SHARD_SIZE == 0:
            shard += 1

    with open(f'{STORAGE_PATH}/cp_dataset.json', 'w') as dataset_metadata:
        data = {
            'shard_size': SHARD_SIZE,
            'shards': shard,
            'length': count,
            'stats': stats
        }
        dataset_metadata.write(json.dumps(data, indent=4))
        print('metadata stored to cp_dataset.json')

    # Report
    print(f'{count} items in the dataset with the following distribution:')
    for key, value in stats.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    main()