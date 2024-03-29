"""
Iterates through the output of the prediction loop from QuackAutoencoder and structures the files for use
as a dataset.
"""
from autoencoder import item_path
from argparse import ArgumentParser
from pathlib import Path
import os
import pickle
import shutil


def main() -> None:
    """
    The reprocessing logic.

    **Required** arguments are:

         `--source_path`
            *str* **Required** The path to top dir of the QuackIterableDataset.
         `--storage_path`
            *str* **Required** The top directory of the data storage tree for the QuackImageDataset.

    Returns
    -------
    void
    """
    # Add args to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--source_path', type=str, required=True)
    arg_parser.add_argument('--storage_path', type=str, required=True)
    args = arg_parser.parse_args()

    # Initialize
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
                metadata['censored'] += 1
            elif item['metadata']['censored'] == 0:
                metadata['undetermined'] += 1
            elif item['metadata']['censored'] == -1:
                metadata['uncensored'] += 1
            # Move:
            shutil.move(file, data_storage)
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
