"""
A helper cli tool for checking files in densnet predictions.
"""
from typing import Iterator
from datetime import datetime
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
from cp_flatten import CensoredPlanetFlatten, Row


def main(args: Namespace):
    """
    Processes possible censorship instances by matching stored metadata with original rows.

    Parameters
    ----------
    args: Namespace
        Valid arguments are:

        --storage_path
            *str*  A path to the top directory of metadata files stored as JSON.
        --

    Returns
    -------
    void
    """
    count = 0
    dates = set()
    storage = Path(args.storage_path)
    files = storage.glob('**/*.json')
    candidates = dict()
    for json_file in files:
        with json_file.open('r') as source:
            item = json.load(source)
        candidates[item['timestamp']] = item
        dates.add(datetime.fromtimestamp(item['timestamp']).date().isoformat())
        count += 1
    print(dates)
    print(count)
    # urls = [
    #     args.source_path
    # ]
    # dataset = CensoredPlanetFlatten(
    #     urls=urls,
    #     compare=True,
    #     labeled=False,
    #     anomalies=True,
    #     raw=True)
    # found = 0
    # dataset: Iterator[Row]
    # for item in dataset:
    #     try:
    #         if candidates[item['start_time']]['ip'] == item['ip'] and candidates[item['start_time']]['domain'] == item['domain']:
    #             candidates[item['start_time']]['row'] = item
    #             found += 1
    #     except KeyError:
    #         continue
    #     if found >= count:
    #         # No need to sift through the rest of the data.
    #         break
    # candidates_storage = storage.joinpath('candidates.json')
    # with candidates_storage.open('w') as target:
    #     json.dump(candidates, target)
    # print(f'Found {count} candidates and wrote {found} augmented candidates to {candidates_storage}')


if __name__ == '__main__':
    # Add args to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--storage_path', type=str, required=True)
    arg_parser.add_argument('--source_path', type=str, required=True)
    args = arg_parser.parse_args()
    main(args)