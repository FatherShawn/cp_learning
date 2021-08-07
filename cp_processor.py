from cp_flatten import CensoredPlanetFlatten
import json
import os
import shutil
import tarfile
import torch

SHARD_SIZE = 3000
STORAGE_PATH = ''
MAX_VERIFICATION_ATTEMPTS = 10

def verify_tarball(path: str):
    verified = False
    if not tarfile.is_tarfile(path):
        # Immediately fails.
        return verified
    with tarfile.open(path, 'r') as tarball:
        try:
            names = tarball.getnames()
            # If we can get names without error the index is likely good.
            # Now verify that all the names read properly.
            for name in names:
                if not 'response-' in name:
                    continue
                response = tarball.extractfile(name)
                item = torch.load(response)

                verify_returned_item(item)
            verified = True
        except tarfile.TarError as e:
            # Corrupted tarfile.
            verified = False
            print(f'Verify failed: {str(e)}')
        except AssertionError as e:
            # One of the assertions in verify_returned_item() failed.
            verified = False
            print(f'Verify failed: {str(e)}')
        except RuntimeError as e:
            # Checking for occurrence of:
            # RuntimeError: PytorchStreamReader failed reading zip archive: too many files
            verified = False
            print(f'Verify failed: {str(e)}')
        except AttributeError as e:
            # Checking for occurrence of:
            # AttributeError: Can't get attribute '_rebwild_tenqor_v2' which is not an actual method.
            # Probably a failed pickle?
            verified = False
            print(f'Verify failed: {str(e)}')
    return verified

def verify_returned_item(item):
    meta = item['metadata']
    assert (isinstance(item, dict)), 'Item from the dataset is not a dictionary.'
    assert ('metadata' in item), 'Key "metadata" not found in item from the dataset.'
    assert ('static_size' in item), 'Key "static_size" not found in item from the dataset.'
    assert ('variable_text' in item), 'Key "variable_text" not found in item from the dataset.'
    assert isinstance(meta, dict)
    assert len(meta) == 5
    assert (isinstance(item['static_size'], torch.Tensor)), 'static_size is not a Tensor'
    assert (isinstance(item['variable_text'], torch.Tensor)), 'variable_text is not a Tensor'
    assert (meta['censored'] in (1, 0, -1)), 'censored value is out of bounds'

def create_shard(id: int) -> None:
    verified = False
    failure_count = 0
    shard_name = f'{STORAGE_PATH}/quack-{id}'
    tarball = f'{STORAGE_PATH}/quack-{id}.tar'
    while not verified:
        shutil.make_archive(shard_name, 'tar', shard_name)
        if verify_tarball(tarball):
            verified = True
            continue
        failure_count += 1
        os.remove(tarball)
        if failure_count > MAX_VERIFICATION_ATTEMPTS:
            print('Max verification failures exceeded.')
            raise RuntimeError
    print(f'verified {tarball}')
    shutil.rmtree(shard_name)


def main() -> None:
    urls = [

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
        meta = item['metadata']
        verify_returned_item(item)
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
            'shard_size': SHARD_SIZE,
            'shards': shard,
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