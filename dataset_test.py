import json
import torch
from cp_dataset import QuackShards



def main():
    # Sample url string: preprocessed/quack-{0..100}.tar
    urls = ''
    dataset = QuackShards(urls)
    count = 0
    stats = {
        'censored': 0,
        'undetermined': 0,
        'uncensored': 0
    }
    for item in dataset:
        assert(isinstance(item, dict)), 'Item from the dataset is not a dictionary.'
        assert('metadata' in item), 'Key "metadata" not found in item from the dataset.'
        assert('static_size' in item), 'Key "static_size" not found in item from the dataset.'
        assert('variable_text' in item), 'Key "variable_text" not found in item from the dataset.'
        meta = json.loads(item['metadata'])
        assert isinstance(meta, dict)
        assert len(meta) == 4
        assert isinstance(item['static_size'], torch.Tensor)
        assert isinstance(item['variable_text'], torch.Tensor)
        assert item['censored'] in (1, 0, -1)
        if item['censored'] == 1:
            stats['censored'] += 1
        elif item['censored'] == 0:
            stats['undetermined'] += 1
        elif item['censored'] == -1:
            stats['uncensored'] += 1
        count += 1
    for key, value in stats.items():
        print(f'{count} items in the dataset with the following distribution:')
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()