from cp_flatten import CensoredPlanetFlatten, TokenizedQuackData
import h5py
from datetime import datetime
import numpy

STORAGE_PATH = '/data/2021-08-16.hdf5'
LOG_PATH = '/data/process_log.txt'

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
            count += 1
            if count % 100000 == 0:
                with open(LOG_PATH, 'a') as log:
                    item_date = datetime.fromtimestamp(meta['timestamp']).date().isoformat()
                    log.write(f'Processed {count:,} items. Last item processed was from {item_date}\n')
        storage.attrs['length'] = count
        storage.attrs['censored'] = stats['censored']
        storage.attrs['undetermined'] = stats['undetermined']
        storage.attrs['uncensored'] = stats['uncensored']

    # Report
    with open(LOG_PATH, 'a') as log:
        log.write(f'{count} items in the dataset with the following distribution:\n')
        for key, value in stats.items():
            log.write(f'{key}: {value}\n')

if __name__ == '__main__':
    main()