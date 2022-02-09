from argparse import ArgumentParser
from cp_dataset import QuackIterableDataset
from cp_flatten_processor import verify_returned_item

def main():
    # Add args to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str, default='/data')
    arg_parser.add_argument('--start', type=int)
    arg_parser.add_argument('--end', type=int)
    args = arg_parser.parse_args()
    dataset = QuackIterableDataset(args.data_dir)
    count = 0
    stats = {
        'censored': 0,
        'undetermined': 0,
        'uncensored': 0
    }
    length = len(dataset)
    start = 0
    end = length
    if args.start is not None and args.end is not None:
        start = args.start
        end = args.end

    for index in range(start, end):
        item = dataset[index]
        # Validate:
        verify_returned_item(item)
        meta = item['metadata']
        if meta['censored'] == 1:
            stats['censored'] += 1
        elif meta['censored'] == 0:
            stats['undetermined'] += 1
        elif meta['censored'] == -1:
            stats['uncensored'] += 1
        count += 1

    print(f'{count} items found in the dataset with the following distribution:')
    for key, value in stats.items():
        print(f'{key}: {value}')

    if args.start is None and args.end is None:
        # Tested the entire dataset
        assert count == len(dataset), f"Dataset should contain{len(dataset)} items but counted {count} items"
        assert stats['censored'] == dataset.censored(), f"Dataset should contain{dataset.censored()} censored items but counted {stats['censored']} items"
        assert stats['undetermined'] == dataset.undetermined(), f"Dataset should contain{dataset.undetermined()} censored items but counted {stats['undetermined']} items"
        assert stats['uncensored'] == dataset.uncensored(), f"Dataset should contain{dataset.uncensored()} censored items but counted {stats['uncensored']} items"
        assert dataset.data_width() > 0, f"Dataset should contain a functional max width"


if __name__ == '__main__':
    main()
