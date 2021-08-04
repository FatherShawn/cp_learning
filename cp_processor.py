from cp_flatten import CensoredPlanetFlatten
import json
import os
import shutil
import torch

SHARD_SIZE = 3000
STORAGE_PATH = '/home/shawn/censored-planet/preprocessed'

def create_shard(id: int) -> None:
    shard_name = f'{STORAGE_PATH}/quack-{id}'
    print(f'archiving {STORAGE_PATH}/quack-{id}')
    shutil.make_archive(shard_name, 'tar', shard_name)
    shutil.rmtree(shard_name)


def main() -> None:
    urls = [
        '/home/shawn/censored-planet/quackDataTar/CP_Quack-echo-2021-07-19-01-01-01.tar.gz'
    ]
    dataset = CensoredPlanetFlatten(urls, True, True)
    count = 0
    shard = 0
    os.makedirs(f'{STORAGE_PATH}/quack-{shard}', 0o755, True)
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
        # Store:
        path = f'{STORAGE_PATH}/quack-{shard}/response-{count % SHARD_SIZE}'
        torch.save(item,path)
        # Count:
        if meta['censored'] == 1:
            stats['censored'] += 1
        elif meta['censored'] == 0:
            stats['undetermined'] += 1
        elif meta['censored'] == -1:
            stats['uncensored'] += 1
        count += 1
        # Check.
        if count % SHARD_SIZE == 0:
            create_shard(shard)
            shard += 1
            os.makedirs(f'{STORAGE_PATH}/quack-{shard}', 0o755, True)
    # Process the last shard. If it has files, it has not been archived.
    unprocessed = sum(len(files) for _, _, files in os.walk(f'{STORAGE_PATH}/quack-{shard}'))
    if (unprocessed):
        create_shard(shard)
    else:
        print(f'{STORAGE_PATH}/quack-{shard} is empty. Removing.')
        shutil.rmtree(f'{STORAGE_PATH}/quack-{shard}')

    with open(f'{STORAGE_PATH}/cp_dataset.json', 'w') as dataset_metadata:
        data = {
            'shards': SHARD_SIZE,
            'length': count,
            'stats': stats
        }
        dataset_metadata.write(json.dumps(data, indent=4))
        print('metadata stored to cp_dataset.json')

    # Report
    print(f'{count} items in the dataset with the following distribution:')
    for key, value in stats.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    main()