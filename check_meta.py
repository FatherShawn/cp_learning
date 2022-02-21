"""
A helper cli tool for checking metadata.pyc files in project datasets.
"""
from argparse import ArgumentParser, Namespace
import pickle

def main(args: Namespace):
    """
    Loads and outputs the keys and values from a metadata.pyc file.

    Parameters
    ----------
    args: Namespace
        Valid arguments are:

        --storage_path
            *str*  A path to the top directory of a dataset.

    Returns
    -------
    void
    """
    with open(args.storage_path + '/metadata.pyc', 'rb') as retrieved_dict:
        metadata = pickle.load(retrieved_dict)
    for key, value in metadata.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    # Add args to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--storage_path', type=str, required=True)
    args = arg_parser.parse_args()
    main(args)