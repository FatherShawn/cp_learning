"""
Iterates through a QuackIterableDataset and creates a QuackImageDataset.
"""
from autoencoder import item_path
from argparse import ArgumentParser
from pathlib import Path
import os
import pickle
import numpy as np


def main() -> None:
    """
    The reprocessing logic.

    **Required** arguments are:

         `--source_path`
            *str* **Required** The path to top dir of the QuackIterableDataset.
         `--storage_path`
            *str* **Required** The top directory of the data storage tree for the QuackImageDataset.

    **Optional** arguments are:
        ` --reduction_factor`
            *float* Probability to include uncensored data.
        ` --filtered`
            *bool* Flag to only include censored and uncensored data.
         `--undetermined`
            *bool* Flag to include only undetermined data
         `--start`
            *int* The starting index in the QuackIterableDataset.
         `--end`
            *int* The ending index in the QuackIterableDataset.

    Returns
    -------
    void
    """
    # Add args to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--source_path', type=str, required=True)
    arg_parser.add_argument('--storage_path', type=str, required=True)
    arg_parser.add_argument('--filtered', action='store_true', default=False)
    arg_parser.add_argument('--evaluate', action='store_true', default=False)
    arg_parser.add_argument('--reduction_factor', type=float)
    args = arg_parser.parse_args()

    # Initialize

    is_filtered = args.filtered and not args.evaluate

    # Prepare to reduce the number of uncensored items.
    rng = np.random.default_rng()

    source_meta = Path(args.source_path + '/metadata.pyc')
    count = 0
    metadata = {
        'censored': 0,
        'undetermined': 0,
        'uncensored': 0,
        'length': 0
    }
    for root, dirs, files in os.walk(args.source_path):
        for file in Path(root).glob('*.pyc'):
            with file.open('rb') as source:
                item = pickle.load(source)
            # Ensure storage is ready.
            storage_path = Path(args.storage_path + item_path(count, dir_only=True))
            storage_path.mkdir(parents=True, exist_ok=True)
            data_storage = Path(args.storage_path + item_path(count, 'pyc'))
            # Count:
            if item['metadata']['censored'] == 1:
                if args.evaluate:
                    continue
                metadata['censored'] += 1
            elif item['metadata']['censored'] == 0:
                if is_filtered:
                    continue
                metadata['undetermined'] += 1
            elif item['metadata']['censored'] == -1:
                if args.evaluate or (is_filtered and rng.random() > args.reduction_factor):
                    # Randomly exclude in proportion to the reduction factor
                    # to keep the data balanced.
                    continue
                metadata['uncensored'] += 1
            # Store:
            with data_storage.open(mode='wb') as target:
                pickle.dump(item, target)
            count += 1
            if count % 10000 == 0:
                print(f'Processed {count:,} items.')
    metadata['length'] = count
    root_meta = Path(args.storage_path + '/metadata.pyc')
    with root_meta.open(mode='wb') as stored_dict:
        pickle.dump(metadata, stored_dict)
    print(f'{count} items re-stored as filtered data.')
    for key, value in metadata.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    main()
