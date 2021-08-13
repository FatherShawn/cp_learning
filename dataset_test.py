import faulthandler
faulthandler.enable()
from cp_dataset import QuackShards
from cp_processor import verify_returned_item
from progress.bar import IncrementalBar



def main():
    url = '/home/shawn/censored-planet/preprocessed'
    dataset = QuackShards(url)
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

    print(f'{count} items in the dataset with the following distribution:')
    for key, value in stats.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()