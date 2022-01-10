import numpy as np

from cp_dataset import QuackIterableDataset
from progress.bar import IncrementalBar
from argparse import ArgumentParser
from pathlib import Path
import pickle
from PIL import Image
from nparray2png import nparray2png
from cp_flatten import QuackConstants


def concatenate_data(item: dict) ->np.ndarray:
    static_source = item['static_size']  # type: np.ndarray
    static_size = []
    variable_text = item['variable_text']  # type: np.ndarray
    # Create an "start marker" XLM-R uses 0, so will we.
    start = np.zeros(1, static_source.dtype)
    # Create an "end marker" XLM-R uses 2, so will we.
    end = np.full(1, 2, static_source.dtype)
    # Build the sequence as a tensor, text first.
    # Time values at static_source index 8 & 9.
    time_values = {8, 9}
    for index in range(static_source.size):
        if index in time_values:
            continue
        # Shift value by vocabulary size to avoid value collisions.
        static_size.append(int(static_source[index] + QuackConstants.VOCAB.value))
    # Now deal with time by finding the difference in seconds.
    time_diff = round((static_source[9] - static_source[8]))
    static_size.append(time_diff + QuackConstants.VOCAB.value)
    return np.concatenate((variable_text, start, np.array(static_size), end), dtype=static_source.dtype).astype(
        np.int_)

def item_path(index: int, suffix:str='png', dir_only:bool=False) -> str:
    rank_five = index // 100000
    remainder = index - (rank_five * 100000)
    rank_three_four = remainder // 1000
    if dir_only:
        return f'/{rank_five}/{rank_three_four}/{index}'
    return f'/{rank_five}/{rank_three_four}/{index}/{index}.{suffix}'


def main() -> None:
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

    with IncrementalBar('Re-mapping', max=length) as bar:
        for index in range(start, end):
            item = dataset[index]
            meta = item['metadata']
            # Ensure storage is ready.
            storage_path = Path(args.storage_path + item_path(count, dir_only=True))
            storage_path.mkdir(parents=True, exist_ok=True)
            image_storage = Path(args.storage_path + item_path(count, 'png'))
            data_storage = Path(args.storage_path + item_path(count, 'pyc'))
            # Count:
            if meta['censored'] == 1:
                metadata['censored'] += 1
            elif meta['censored'] == 0:
                if args.filtered:
                    bar.next()
                    continue
                else:
                    metadata['undetermined'] += 1
            elif meta['censored'] == -1:
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
            bar.next()
    metadata['length'] = count
    with root_meta.open(mode='wb') as stored_dict:
        pickle.dump(metadata, stored_dict)
    print(f'{count} items re-stored as image and pickled data')
    for key, value in metadata.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    main()
