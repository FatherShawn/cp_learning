import json
import tarfile
import re
import logging
import torch
import geoip2.database
import geoip2.errors
from blockpage import BlockpageMatcher
from collections import defaultdict
from dateutil.parser import isoparse
from io import BufferedReader
from torch.utils.data import IterableDataset
from typing import Any, Callable, Dict, Iterator, Set, Tuple, TypedDict, Union
from urlextract import URLExtract
from webdataset import ShardList, Shorthands, tariterators, url_opener

# Constants

SATELLITE_TAGS = {'ip', 'http', 'asnum', 'asname', 'cert'}
INTERFERENCE_IPDOMAIN: Dict[str, Set[str]] = defaultdict(set)

# For Hyperquack v1
# echo/discard domain and url content
SENT_PATTERN = "GET (.*) HTTP/1.1\r\nHost: (.*)\r\n"

# For Hyperquack v1
CONTROL_URLS = [
    'example5718349450314.com',  # echo/discard
    'rtyutgyhefdafioasfjhjhi.com'  # HTTP/S
]

class Row(TypedDict):
    """
    A data structure for a single flattened row of CP data.

    If data is labeled:
    censorship: 1 => definitely, 0 => unknown, -1 => definitely not

    If data is unlabeled, censorship defaults to 0.
    """
    ip: str
    domain: str
    anomaly: bool
    controls_failed: bool
    stateful_block: bool
    success: bool
    error: str
    start_time: float
    end_time: float
    censored: int
    received_tls_version: int
    received_tls_cipher_suite: int
    received_tls_cert: str
    sent: str
    received_status: str
    received_headers: str
    received_body: str

class MetaTensor(TypedDict):
    metadata: str
    censored: int
    static_size: torch.Tensor
    variable_text: torch.Tensor

class CensoredPlanetFlatten(IterableDataset, Shorthands):
    """
    Although (Webdataset)[https://webdataset.github.io/webdataset/] may be able to handle all our pipeline needs,
    my intention here is to take in the Censored Planet Quack data and pre-preprocess it into Pytorch Tensors.

    The following are adapted from https://github.com/censoredplanet/censoredplanet-analysis/blob/master/pipeline/metadata/flatten.py
    - process_hyperquack_v1
    - process_hyperquack_v2
    - extract_domain_from_sent_field

    Parameters
    ----------
    self.__shards: ShardList
        The dataset to use a pipeline source.
    self.__blockpage_matcher: BlockpageMatcher
        The blockpage matching utility.
    self.__labeled: bool
        Should the data be labeled as censored?
    self.__xlmr: object
        The XLMR pretrained tokenizer.
    """

    def __init__(self, urls: Union[str, list[str]], labeled:bool = False, anomalies:bool = False) -> None:
        super().__init__()

        assert (
                urls is not None
        ), "Must supply a url as a string or list of strings"

        self.__shards = ShardList(urls)
        self.__blockpage_matcher = BlockpageMatcher()
        self.__labeled = labeled
        self.__anomalies = anomalies
        self.__ip2geo = geoip2.database.Reader('./mmdb/country.mmdb')
        self.__xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
        self.__xlmr.eval()

    def __iter__(self) -> Iterator[torch.Tensor]:
        for quack_file in url_opener(self.__shards):
            for filestream in self.__quack_file_expander(quack_file):
                file_name, connection = filestream
                iterate_lines = True
                while iterate_lines:
                    try:
                        # Get the next line.
                        line = connection.readline()
                        if line == b'':
                            # End of file.
                            raise StopIteration
                        # Flatten the line.
                        try:
                            scan = json.loads(line.decode('utf-8', errors='replace'))
                        except json.decoder.JSONDecodeError as e:
                            logging.warning('JSONDecodeError: %s\nFilename: %s\n%s\n', e, file_name,
                                            line)
                            continue
                        if 'Server' in scan:
                            try:
                                blocked = scan['Blocked']
                            except KeyError:
                                blocked = False
                            if self.__anomalies and not blocked:
                                continue
                            yield from self.__process_hyperquack_v1(scan)
                        elif 'vp' in scan:
                            try:
                                blocked = scan['anomaly']
                            except KeyError:
                                blocked = False
                            if self.__anomalies and not blocked:
                                continue
                            yield from self.__process_hyperquack_v2(scan)
                        else:
                            raise Exception(f"Line with unknown hyperquack format:\n{scan}")
                    except StopIteration:
                        iterate_lines = False
                    except Exception as exn:
                        tariterators.reraise_exception(exn)


    # Utility iterators to keep __iter__ readable.

    def __quack_tar_file_iterator(self, file_obj: BufferedReader,
                                  handler: Callable = tariterators.reraise_exception) -> Tuple[str, object]:
        """
        An adaptation of webdataset.tariterators.tar_file_expander that returns a stream to a file in which we are
        interested rather than reading the entire file.

        Parameters
        ----------
        file_obj : BufferedReader
            A tarfile stream to the tar file.
        skip_meta : str
            A regular expression for file names to skip.
        handler: Callable
            The exception handler.

        Yields
        -------
        tuple
            A tuple with 2 members:
            - str: filename
            - tarfile._Stream: An internal class used by the tarfile module

        """
        stream = tarfile.open(fileobj=file_obj, mode="r|*")
        for tarinfo in stream:
            try:
                if not tarinfo.isreg():
                    continue
                file_name = tarinfo.name
                if file_name is None:
                    continue
                if not 'results.json' in file_name:
                    # Only interested in results.json
                    continue
                yield file_name, stream.extractfile(tarinfo)
            except Exception as exn:
                if handler(exn):
                    continue
                else:
                    break
        del stream

    def __quack_file_expander(self, data: Dict[str, BufferedReader],
                              handler: Callable = tariterators.reraise_exception) -> Iterator[Tuple[str, object]]:
        """
        An adaptation of webdataset.tariterators.tar_file_expander that calls our quack_tar_file_iterator
        instead.

        Parameters
        ----------
        data : dict
            A dictionary with stream and filename keys.
        handler: Callable
            The exception handler.

        Yields
        -------
        tuple
            A tuple with 2 members:
            - str: filename
            - tarfile._Stream: An internal class used by the tarfile module
        """
        try:
            assert "stream" in data
            for sample in self.__quack_tar_file_iterator(data["stream"]):
                assert isinstance(sample, tuple) and len(sample) == 2
                yield sample
        except Exception as exn:
            handler(exn)

    def __process_hyperquack_v1(self, scan: Dict) -> Iterator[MetaTensor]:
        """
        Process a line of Echo/Discard/HTTP/S data in HyperQuack V1 format.

        https://censoredplanet.readthedocs.io/en/latest/hyperquackv1.html

        Parameters
        ----------
        filename: str
            A filepath string
        scan: dict
            A loaded json object containing the parsed content of the line

        Yields
        -------
        MetaTensor
        """
        for index, result in enumerate(scan.get('Results', [])):
            domain = self.__extract_domain_from_sent_field(result['Sent'])
            is_control = domain in CONTROL_URLS
            # Due to a bug the sent field sometimes isn't populated
            # when the measurement failed due to network timeout.
            if not domain:
                # Control measurements come at the end, and are not counted as retries.
                is_control = index > scan['Retries']
                if not is_control:
                    domain = scan['Keyword']
            if is_control:
                # We are not interested in control queries.
                continue

            error = ''
            if 'Error' in result:
                error = result['Error']
            received = result.get('Received', '')
            received_fields = self.__parse_received_data(received)
            # Calculate censorship if required
            matches_blockpage = 0
            if self.__labeled and result['Success'] and not scan['Blocked']:
                matches_blockpage = -1
            if self.__labeled and len(received_fields['received_body']) > 0:
                matches_blockpage = self.__blockpage_match(received_fields['received_body'])

            # Process time strings into a unix timestamp.
            start = isoparse(result['StartTime'])
            end = isoparse(result['EndTime'])

            # Create row data.
            row = Row(
                ip=scan['Server'],
                domain=domain,
                anomaly=scan['Blocked'],
                censored=matches_blockpage,
                controls_failed=scan['FailSanity'],
                stateful_block=scan['StatefulBlock'],
                success=result['Success'],
                error=error,
                start_time=start.timestamp(),
                end_time=end.timestamp(),
                sent=result['Sent'],
                received_tls_version=received_fields['received_tls_version'],
                received_tls_cipher_suite=received_fields['received_tls_cipher_suite'],
                received_tls_cert=received_fields['received_tls_cert'],
                received_status=received_fields['received_status'],
                received_headers=received_fields['received_headers'],
                received_body=received_fields['received_body']
            )
            meta_tensor = self.__process_row(row)
            yield  meta_tensor

    def __process_hyperquack_v2(self, scan: Dict) -> Iterator[MetaTensor]:
        """
        Process a line of Echo/Discard/HTTP/S data in HyperQuack V2 format.

        https://censoredplanet.readthedocs.io/en/latest/hyperquackv2.html

        Parameters
        ----------
        filename: str
            A filepath string
        scan: dict
            A loaded json object containing the parsed content of the line

        Yields
        -------
        MetaTensor
        """
        controls_failed = False
        if 'controls_failed' in scan:
            controls_failed = scan['controls_failed'] == True

        for response in scan.get('response', []):
            if 'control_url' in response:
                # We are not interested in control queries.
                continue
            error = ''
            if 'error' in response:
                error = response['error']
            received = response.get('response', '')
            received_fields = self.__parse_received_data(received)
            # Calculate censorship if required
            matches_blockpage = 0
            if self.__labeled and response['matches_template'] and not scan['anomaly']:
                matches_blockpage = -1
            if self.__labeled and 'body' in received:
                matches_blockpage = self.__blockpage_match(received['body'])


            # Process time strings into a unix timestamp.
            start = isoparse(response['start_time'])
            end = isoparse(response['end_time'])

            # Create row data.
            row = Row(
                ip=scan['vp'],
                domain=scan['test_url'],
                anomaly=scan['anomaly'],
                censored=matches_blockpage,
                controls_failed=controls_failed,
                stateful_block=scan['stateful_block'],
                success=response['matches_template'],
                error=error,
                start_time=start.timestamp(),
                end_time=end.timestamp(),
                sent=scan['test_url'],
                received_tls_version=received_fields['received_tls_version'],
                received_tls_cipher_suite=received_fields['received_tls_cipher_suite'],
                received_tls_cert=received_fields['received_tls_cert'],
                received_status=received_fields['received_status'],
                received_headers=received_fields['received_headers'],
                received_body=received_fields['received_body']
            )
            yield self.__process_row(row)

    def __process_satellite(self, filename: str, scan: Dict) -> MetaTensor:
        """

        Parameters
        ----------
        filename: str
        scan: dict

        Returns
        -------
        MetaTensor
        """
        pass

    def __process_row(self, row: Row) -> MetaTensor:
        """
        Transforms flattened data in a Row into a torch.Tensor with metadata.

        Parameters
        ----------
        row: Row

        Returns
        -------
        MetaTensor
        """
        # See if we can look up a country from the ip.

        try:
            lookup = self.__ip2geo.country(row['ip'])
            country = lookup.country.name
        except geoip2.errors.AddressNotFoundError:
            country = None

        metadata = {
            'domain': row['domain'],
            'ip': row['ip'],
            'location': country,
            'timestamp': row['start_time']
        }
        # Row keys with static length data.
        static_keys = ('success', 'anomaly', 'controls_failed', 'stateful_block', 'start_time', 'end_time', 'received_tls_version', 'received_tls_cipher_suite', 'received_tls_cert')
        # Row keys with variable length (text) data.
        ##> Skipping 'received_tls_cert' for now.
        text_keys = ('sent', 'received_status', 'received_status', 'received_headers', 'received_body')
        static_dimension = []
        # First split the ip and cast to int.
        for segment in row['ip'].split('.'):
            static_dimension.append(int(segment))
        for key in static_keys:
            static_dimension.append(row[key])
        concatenated = ''
        # Concatenate the strings.
        for key in text_keys:
            concatenated += row[key]
        meta_tensor = MetaTensor(
            metadata=json.dumps(metadata),
            censored=row['censored'],
            static_size=torch.tensor(static_dimension),
            variable_text=self.__xlmr.encode(concatenated)
        )
        return meta_tensor

    def __extract_domain_from_sent_field(self, sent: str) -> str:
        """
        Get the url out of a 'sent' field in a measurement.

        Parameters
        ----------
        sent: str

            "" meaning the sent packet wasn't recorded.
            "GET / HTTP/1.1\r\nHost: example5718349450314.com\r\n" (echo/discard)
            "GET www.bbc.co.uk HTTP/1.1\r\nHost: /content.html\r\n" (discard error)
            or just "www.apple.com" (HTTP/S)

        Returns
        -------
        str
            Just the url, if found.
        """
        extractor = URLExtract()
        extractor.update_when_older(7)  # updates known TLD when list is older that 7 days

        if sent == '':
            return sent

        match = re.search(SENT_PATTERN, sent)
        if match:
            path = match.group(1)
            domain = match.group(2)

            # This is a bug where the domain and path were reversed in content sent.
            # We do our best to reconstruct the intended url
            # by swapping them to their intended position
            if extractor.has_urls(path):
                domain, path = path, domain

            if path == '/':
                return domain
            return domain + path

        if ' ' not in sent:
            return sent

        raise Exception(f"unknown sent field format: {sent}")

    def __parse_received_data(self, received: Union[str, Dict[str, Any]]) -> Dict:
        """

        Parameters
        ----------
        received: Union[str, dict[str, Any]]
            A dict parsed from json data, or a str

        Returns
        -------
        dict

        """
        data = {
            'received_status': '',
            'received_body': '',
            'received_headers': '',
            'received_tls_version': 0,
            'received_tls_cipher_suite': 0,
            'received_tls_cert': 0,
        }
        if isinstance(received, str):
            data['received_status'] = received
            return data
        if 'status_line' in received:
            data['received_status'] = received['status_line']
        if 'body' in received:
            data['received_body'] = received['body']
        if 'headers' in received:
            for key, values in received['headers'].items():
                for value in values:
                    data['received_headers'] += key + ': ' + value + ' '
        # hyperquack v1 TLS format
        tls = received.get('tls', None)
        if tls:
            data['received_tls_version'] = tls['version']
            data['received_tls_cipher_suite'] = tls['cipher_suite']
            data['received_tls_cert'] = tls['cert']

        # hyperquack v2 TLS format
        if 'TlsVersion' in received:
            data['received_tls_version'] = received['TlsVersion']
            data['received_tls_cipher_suite'] = received['CipherSuite']
            data['received_tls_cert'] = received['Certificate']
        return data

    def __blockpage_match(self, body) -> int:
        """

        Parameters
        ----------
        body: str
        An html body string

        Returns
        -------
        int
            Returns -1, 0 or 1.
            * 1 if a match to a known block page.
            * -1 if a match to a known false positive page.
            * 0 if no match.
        """
        blockpage, signature = self.__blockpage_matcher.match_page(body)
        # The matcher returns True, False or None.
        # True if a match to a known block page.
        # False if a match to a known false positive page.
        # None no match at all.
        # Translating to 3 integer values for type consistency.
        if blockpage is None:
            return 0
        if blockpage:
            return 1
        return -1