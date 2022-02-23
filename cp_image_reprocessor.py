"""
Iterates through a QuackIterableDataset and creates a QuackImageDataset.
"""
from autoencoder import item_path
from cp_tokenized_data import concatenate_data
from cp_dataset import QuackIterableDataset
from argparse import ArgumentParser
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from nparray2png import nparray2png


def main() -> None:
    """
    The reprocessing logic.

    **Required** arguments are:

         `--source_path`
            *str* **Required** The path to top dir of the QuackIterableDataset.
         `--storage_path`
            *str* **Required** The top directory of the data storage tree for the QuackImageDataset.
        ` --filtered`
            *bool* Flag to only include censored and uncensored data.
         `--undetermined`
            *bool* Flat to include only undetermined data
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
    arg_parser.add_argument('--start', type=int)
    arg_parser.add_argument('--end', type=int)
    args = arg_parser.parse_args()

    dataset = QuackIterableDataset(args.source_path)
    length = len(dataset)
    start = 0
    end = length
    root_meta = Path(args.storage_path + '/metadata.pyc')
    if args.start is not None and args.end is not None:
        start = args.start
        end = args.end
        length = end - start

    # Prepare to reduce the number of uncensored items.
    rng = np.random.default_rng()
    source_meta = Path(args.source_path + '/metadata.pyc')
    try:
        with source_meta.open(mode='rb') as retrieved_dict:
            source_metadata = pickle.load(retrieved_dict)
        reduction_factor = source_metadata['uncensored'] / source_metadata['censored']
    except (OSError, KeyError):
        reduction_factor = 1
    print(f'Reduction factor is {reduction_factor:4f}.')

    try:
        with root_meta.open(mode='rb') as retrieved_dict:
            metadata = pickle.load(retrieved_dict)
            count = metadata['length']
    except OSError:
        count = 0
        metadata = {
            'censored': 0,
            'undetermined': 0,
            'uncensored': 0,
            'length': 0
        }

    for index in range(start, end):
        item = dataset[index]
        meta = item['metadata']
        # Ensure storage is ready.
        storage_path = Path(args.storage_path + item_path(count, dir_only=True, is_collection=True))
        storage_path.mkdir(parents=True, exist_ok=True)
        image_storage = Path(args.storage_path + item_path(count, 'png', is_collection=True))
        data_storage = Path(args.storage_path + item_path(count, 'pyc', is_collection=True))
        # Count:
        if meta['censored'] == 1:
            metadata['censored'] += 1
        elif meta['censored'] == 0:
            if args.filtered:
                continue
            metadata['undetermined'] += 1
        elif meta['censored'] == -1:
            if args.filtered and rng.random() > reduction_factor:
                # Randomly exclude in proportion to the reduction factor
                # to keep the data balanced.
                continue
            metadata['uncensored'] += 1
        # Store:
        pixels = nparray2png(concatenate_data(item))
        data = {
            'metadata': meta,
            'pixels': pixels
        }
        with data_storage.open(mode='wb') as target:
            pickle.dump(data, target)
        generated_image = Image.fromarray(pixels, mode='L')
        generated_image.save(image_storage)
        count += 1
        if count % 10000 == 0:
            print(f'Processed {count:,} items.')
    metadata['length'] = count
    with root_meta.open(mode='wb') as stored_dict:
        pickle.dump(metadata, stored_dict)
    print(f'{count} items re-stored as image and pickled data')
    for key, value in metadata.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    main()
