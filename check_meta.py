from argparse import ArgumentParser
import pickle

if __name__ == '__main__':
    # Add args to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--storage_path', type=str, required=True)
    args = arg_parser.parse_args()
    with open(args.storage_path + '/metadata.pyc', 'rb') as retrieved_dict:
        metadata = pickle.load(retrieved_dict)
    for key, value in metadata.items():
        print(f'{key}: {value}')