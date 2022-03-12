"""
A helper cli tool for checking files in densnet predictions.
"""
from typing import Iterator
from datetime import datetime
from argparse import ArgumentParser, Namespace
import json
import re
from pathlib import Path
from cp_flatten import CensoredPlanetFlatten, Row


def main(args: Namespace):
    """
    Processes possible censorship instances by matching stored metadata with original rows and sorting into types.

    Parameters
    ----------
    args: Namespace
        Valid arguments are:

        --storage_path
            *str*  A path to the top directory of metadata files stored as JSON.
        --source_path
            *str*  A path to tar file of original Censored Planet data.

    Returns
    -------
    void
    """
    count = 0
    storage = Path(args.storage_path)
    files = storage.glob('**/*.json')
    candidates = dict()
    for json_file in files:
        with json_file.open('r') as source:
            item = json.load(source)
        candidates[item['timestamp']] = item
        count += 1
    print(count)
    urls = [
        args.source_path
    ]
    dataset = CensoredPlanetFlatten(
        urls=urls,
        compare=True,
        labeled=False,
        anomalies=True,
        raw=True)
    found = 0
    dataset: Iterator[Row]
    for item in dataset:
        try:
            if candidates[item['start_time']]['ip'] == item['ip'] and candidates[item['start_time']]['domain'] == item['domain']:
                candidates[item['start_time']]['row'] = item
                found += 1
        except KeyError:
            continue
        if found >= count:
            # No need to sift through the rest of the data.
            break
    candidates_storage = storage.joinpath('candidates.json')
    with candidates_storage.open('w') as target:
        json.dump(candidates, target)
    print(f'Found {count} candidates and wrote {found} augmented candidates to {candidates_storage}')

    # Analyze the augmented candidates and output a summary.

    # Regular expressions developed by repeatedly reviewing the `candidates.json` file for patterns.
    bitdefender = re.compile('<title>Bitdefender Alert Page</title>|connect.bitdefender.net')
    seqrite = re.compile('<title>Seqrite Endpoint Security</title>')
    xtra = re.compile('Extra Veilig Internet heeft deze website geblokkeerd')
    net_protector = re.compile('Net Protector::Web Protection')
    att = re.compile('myhomenetwork.att.com')
    meraki = re.compile(r'meraki.com:\d+/blocked.cgi')
    reset = re.compile('connection reset by peer')
    code_302 = re.compile(r'HTTP/1.\d 302')
    closed = re.compile('Connection: close')
    no_route = re.compile('no route to host')
    io_timeout = re.compile('i/o timeout')
    connection_timeout = re.compile('connection timed out')
    refused = re.compile('connection refused')
    no_data = re.compile('does not match echo request')
    unreachable = re.compile('network is unreachable')

    # Counter dictionary.
    type = {
        'bitdefender': {
            'count': 0,
            'locations': set()
        },
        'meraki': {
            'count': 0,
            'locations': set()
        },
        'xtra': {
            'count': 0,
            'locations': set()
        },
        'net_protect': {
            'count': 0,
            'locations': set()
        },
        'att': {
            'count': 0,
            'locations': set()
        },
        'seqrite': {
            'count': 0,
            'locations': set()
        },
        'reset': {
            'count': 0,
            'locations': set()
        },
        '302': {
            'count': 0,
            'locations': set(),
        },
        'no_route': {
            'count': 0,
            'locations': set(),
        },
        'io_timeout': {
            'count': 0,
            'locations': set(),
        },
        'connect_timeout': {
            'count': 0,
            'locations': set(),
        },
        'refused': {
            'count': 0,
            'locations': set(),
        },
        'closed': {
            'count': 0,
            'locations': set(),
        },
        'unreachable': {
            'count': 0,
            'locations': set(),
        },
        'no_data': {
            'count': 0,
            'locations': set(),
        },
        'other': {
            'count': 0,
            'locations': set()
        },
    }
    unknown = []

    # Iterate and filter.
    for value in candidates.values():
        if bitdefender.search(value['row']['received_body']) is not None:
            type['bitdefender']['count'] += 1
            type['bitdefender']['locations'].add(value['location'])
        elif seqrite.search(value['row']['received_body']) is not None:
            type['seqrite']['count'] += 1
            type['seqrite']['locations'].add(value['location'])
        elif net_protector.search(value['row']['received_body']) is not None:
            type['net_protect']['count'] += 1
            type['net_protect']['locations'].add(value['location'])
        elif xtra.search(value['row']['received_body']) is not None:
            type['xtra']['count'] += 1
            type['xtra']['locations'].add(value['location'])
        elif att.search(value['row']['received_body']) is not None:
            type['att']['count'] += 1
            type['att']['locations'].add(value['location'])
        elif meraki.search(value['row']['received_body']) is not None:
            type['meraki']['count'] += 1
            type['meraki']['locations'].add(value['location'])
        elif code_302.search(value['row']['received_body']) is not None:
            type['302']['count'] += 1
            type['302']['locations'].add(value['location'])
        elif closed.search(value['row']['received_body']) is not None:
            type['closed']['count'] += 1
            type['closed']['locations'].add(value['location'])
        elif reset.search(value['row']['error']) is not None:
            type['reset']['count'] += 1
            type['reset']['locations'].add(value['location'])
        elif connection_timeout.search(value['row']['error']) is not None:
            type['connect_timeout']['count'] += 1
            type['connect_timeout']['locations'].add(value['location'])
        elif no_route.search(value['row']['error']) is not None:
            type['no_route']['count'] += 1
            type['no_route']['locations'].add(value['location'])
        elif io_timeout.search(value['row']['error']) is not None:
            type['io_timeout']['count'] += 1
            type['io_timeout']['locations'].add(value['location'])
        elif unreachable.search(value['row']['error']) is not None:
            type['unreachable']['count'] += 1
            type['unreachable']['locations'].add(value['location'])
        elif refused.search(value['row']['error']) is not None:
            type['refused']['count'] += 1
            type['refused']['locations'].add(value['location'])
        elif (no_data.search(value['row']['error']) is not None) and (len(value['row']['received_body']) == 0):
            type['no_data']['count'] += 1
            type['no_data']['locations'].add(value['location'])
        else:
            unknown.append(value)
    if len(unknown) > 0:
        candidates_storage = storage.joinpath('other_candidates.json')
        with candidates_storage.open('w') as target:
            json.dump(unknown, target, indent=2)

    # Print with latex table formatting codes.
    print(f"Bitdefender Alert Page blockpage & {type['bitdefender']['count']} & {', '.join([str(location) for location in sorted(type['bitdefender']['locations'])])} \\\\")
    print(f"Seqrite Endpoint Security blockpage & {type['seqrite']['count']} & {', '.join([str(location) for location in sorted(type['seqrite']['locations'])])} \\\\")
    print(f"Meraki blockpage & {type['meraki']['count']} & {', '.join([str(location) for location in sorted(type['meraki']['locations'])])} \\\\")
    print(f"Net Protector blockpage & {type['net_protect']['count']} & {', '.join([str(location) for location in sorted(type['net_protect']['locations'])])} \\\\")
    print(f"ATT blockpage & {type['att']['count']} & {', '.join([str(location) for location in sorted(type['att']['locations'])])} \\\\")
    print(f"Extra Safe Internet blockpage & {type['xtra']['count']} & {', '.join([str(location) for location in sorted(type['xtra']['locations'])])} \\\\")
    print(f"HTTP code 302 & {type['302']['count']} & {', '.join([str(location) for location in sorted(type['302']['locations'])])} \\\\")
    print(f"`Connection: close` message returned without blockpage & {type['closed']['count']} & {', '.join([str(location) for location in sorted(type['closed']['locations'])])} \\\\")
    print(f"`connection reset by peer' error & {type['reset']['count']} & {', '.join([str(location) for location in sorted(type['reset']['locations'])])} \\\\")
    print(f"`no route to host' error & {type['no_route']['count']} & {', '.join([str(location) for location in sorted(type['no_route']['locations'])])} \\\\")
    print(f"`connection timed out' error & {type['connect_timeout']['count']} & {', '.join([str(location) for location in sorted(type['connect_timeout']['locations'])])} \\\\")
    print(f" `i/o timeout' error & {type['io_timeout']['count']} & {', '.join([str(location) for location in sorted(type['io_timeout']['locations'])])} \\\\")
    print(f"`network is unreachable' error & {type['unreachable']['count']}  & {', '.join([str(location) for location in sorted(type['unreachable']['locations'])])} \\\\")
    print(f"`connection refused' error & {type['refused']['count']} & {', '.join([str(location) for location in sorted(type['refused']['locations'])])} \\\\")
    print(f"`echo response does not match echo request' and no other data & {type['no_data']['count']} & {', '.join([str(location) for location in sorted(type['io_timeout']['locations'])])} \\\\")
    print(f"Found {len(unknown)} other type of candidates")


if __name__ == '__main__':
    # Add args to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--storage_path', type=str, required=True)
    arg_parser.add_argument('--source_path', type=str, required=True)
    args = arg_parser.parse_args()
    main(args)