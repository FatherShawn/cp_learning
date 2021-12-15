from argparse import ArgumentParser
from cp_dataset import QuackIterableDataset
from cp_processor import verify_returned_item
from progress.bar import IncrementalBar


def main():
    # Add args to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str, default='/data')
    args = arg_parser.parse_args()
    dataset = QuackIterableDataset(args.data_dir)
    count = 0
    stats = {
        'censored': 0,
        'undetermined': 0,
        'uncensored': 0
    }

    with IncrementalBar('Verifying', max=len(dataset)) as bar:
        for item in dataset:
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
            bar.next()

    print(f'{count} items found in the dataset with the following distribution:')
    for key, value in stats.items():
        print(f'{key}: {value}')

    assert count == len(dataset), f"Dataset should contain{len(dataset)} items but counted {count} items"
    assert stats['censored'] == dataset.censored(), f"Dataset should contain{dataset.censored()} censored items but counted {stats['censored']} items"
    assert stats['undetermined'] == dataset.undetermined(), f"Dataset should contain{dataset.undetermined()} censored items but counted {stats['undetermined']} items"
    assert stats['uncensored'] == dataset.uncensored(), f"Dataset should contain{dataset.uncensored()} censored items but counted {stats['uncensored']} items"
    assert dataset.data_width() > 0, f"Dataset should contain a functional max width"


if __name__ == '__main__':
    main()
