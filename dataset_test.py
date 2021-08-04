import torch
from cp_dataset import QuackShards



def main():
    # faulthandler.enable()
    # Sample url string: preprocessed/quack-{0..100}.tar
    url = '/home/shawn/censored-planet/preprocessed'
    dataset = QuackShards(url)
    count = 0
    stats = {
        'censored': 0,
        'undetermined': 0,
        'uncensored': 0
    }
    for item in dataset:
        # Validate:
        assert (isinstance(item, dict)), 'Item from the dataset is not a dictionary.'
        assert ('metadata' in item), 'Key "metadata" not found in item from the dataset.'
        assert ('static_size' in item), 'Key "static_size" not found in item from the dataset.'
        assert ('variable_text' in item), 'Key "variable_text" not found in item from the dataset.'
        meta = item['metadata']
        assert isinstance(meta, dict)
        assert len(meta) == 5
        assert (isinstance(item['static_size'], torch.Tensor)), 'static_size is not a Tensor'
        assert (isinstance(item['variable_text'], torch.Tensor)), 'variable_text is not a Tensor'
        assert (meta['censored'] in (1, 0, -1)), 'censored value is out of bounds'
        if meta['censored'] == 1:
            stats['censored'] += 1
        elif meta['censored'] == 0:
            stats['undetermined'] += 1
        elif meta['censored'] == -1:
            stats['uncensored'] += 1
        count += 1
    print(f'{count} items in the dataset with the following distribution:')
    for key, value in stats.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()