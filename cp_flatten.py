"""
CensoredPlanetFlatten and helper classes to flatten raw Censored Planet data.
"""
import json
import tarfile
import re
import logging
import geoip2.database
import geoip2.errors
import numpy as np
import pickle
from enum import Enum
from torch.utils.data.dataset import T_co
from blockpage import BlockpageMatcher
from dateutil.parser import isoparse
from fairseq.models.roberta import XLMRModel
from io import BufferedReader
from torch.utils.data import IterableDataset
from typing import Any, Callable, Dict, Iterator, Tuple, TypedDict, Union, List
from urlextract import URLExtract
from webdataset import ShardList, Shorthands, tariterators, url_opener


class QuackConstants(Enum):
    """
    An Enum to contain constants used in this project.
    """

    # For Hyperquack v1
    # echo/discard domain and url content
    SENT_PATTERN = "GET (.*) HTTP/1.1\r\nHost: (.*)\r\n"  # type: str

    # For Hyperquack v1
    CONTROL_URLS = [
        'example5718349450314.com',  # echo/discard
        'rtyutgyhefdafioasfjhjhi.com'  # HTTP/S
    ]  # type: List[str]
    # Re-mapped XLMR tokens in use to a smaller vocab:
    VOCAB = 6813
    # XLM-R uses 1 as the token for <pad>.
    XLMR_PAD = 1  # type: int
    # All data falls after July 1, 2021:
    TIME_FLOOR = isoparse('2021-07-01').timestamp()  # type: float
    # All data falls within a single year:
    TIME_CEILING = isoparse('2022-07-01').timestamp()  # type: float


class Row(TypedDict):
    """
    A data structure for a single flattened row of CP data.

    If data is labeled:
    censorship: 1 => definitely, 0 => unknown, -1 => definitely not

    If data is unlabeled, censorship defaults to 0.
    """
    ip: str
    location: str
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


class TokenizedQuackData(TypedDict):
    """
    A data structure to hold the flattened data.
    """
    metadata: dict
    static_size: np.ndarray
    variable_text: np.ndarray


class CensoredPlanetFlatten(IterableDataset, Shorthands):
    """
    Although (Webdataset)[https://webdataset.github.io/webdataset/] may be able to handle all our pipeline needs,
    my intention here is to take in the Censored Planet Quack data and pre-preprocess it into Pytorch Tensors.

    The following are adapted from https://github.com/censoredplanet/censoredplanet-analysis/blob/master/pipeline/metadata/flatten.py

    - `process_hyperquack_v1`
    - `process_hyperquack_v2`
    - `extract_domain_from_sent_field`
    """


    def __init__(self,
                 urls: Union[str, List[str]],
                 vocab_path: str = '',
                 compare: bool = False,
                 labeled: bool = False,
                 anomalies: bool = False,
                 raw: bool = False
                 ) -> None:
        """

        Parameters
        ----------
        urls: Union[str, List[str]]
            Path or paths to pass to webdataset.dataset.ShardList. Points to Censored Planet .tar data files.
        vocab_path: str
            Path to a .pyc file which holds a dictionary that maps an index sequence with tokens used from
            fairseq.models.roberta.model_xlmr.XLMRModel when flattening data.
        compare: bool
            Should data be compared with Censored Planet blockpage signatures?
        labeled: bool
            Should only data successfully precessed by blockpage matcher be returned?
        anomalies: bool
            Should only data marked by Censored Planet as an anomaly be processed?
        raw: bool
            Should the raw row be returned without processing into vectors?
        """
        super().__init__()

        assert (
                urls is not None
        ), "Must supply a url as a string or list of strings"

        self.__shards = ShardList(urls)
        self.__blockpage_matcher = BlockpageMatcher()
        self.__labeled = labeled
        self.__compare = labeled or compare
        self.__anomalies = anomalies
        self.__raw = raw
        if not self.__raw:
            # Bring in the MMDB free database.
            self.__ip2geo = geoip2.database.Reader('./mmdb/country.mmdb')
            # Bring in the pretrained XLMR model.
            self.__xlmr = XLMRModel.from_pretrained('/data/xlmr.large', checkpoint_file='model.pt')
            self.__xlmr.eval()
            self.__vocab_path = vocab_path
            try:
                with open(vocab_path, 'rb') as retrieved_dict:
                    self.__vocab = pickle.load(retrieved_dict)
            except OSError:
                self.__vocab = dict()
            self.__vocab_next = len(self.__vocab)


    def __getitem__(self, index) -> T_co:
        """
        Required by the parent of IterableDataset but not useful in this context, and not implemented by any of the
        Webdataset implementations of IterableDataset.
        """
        pass

    def __iter__(self) -> Iterator[Union[TokenizedQuackData, Row]]:
        """
        Iterates the data in the .tar files.

        Returns
        -------
        Union[Iterator[TokenizedQuackData], Iterator[Row]]
            A dictionary (TypedDict) containing flattened data for a single item or if self.__raw is true, the
            unprocessed (Row) dictionary of row data is returned.
        """
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
                            logging.warning(f'JSONDecodeError: {e}\nFilename: {file_name}\n')
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
                            print(f"Line skipped with unknown hyperquack format:\n{scan}")
                    except StopIteration:
                        iterate_lines = False
                    except Exception as exn:
                        tariterators.reraise_exception(exn)
        if not self.__raw:
            # Save the xlmr -> vocab token mapping:
            with open(self.__vocab_path, 'wb') as stored_dict:
                pickle.dump(self.__vocab, stored_dict)
            print(f"All items flattened.  Re-mapped vocabulary has {len(self.__vocab)} items.")

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

    def __process_hyperquack_v1(self, scan: Dict) -> Union[Iterator[TokenizedQuackData], Iterator[Row]]:
        """
        Process a line of Echo/Discard/HTTP/S data in HyperQuack V1 format.

        Parameters
        ----------
        scan: dict
            A loaded json object containing the parsed content of the line

        Yields
        -------
        Union[Iterator[TokenizedQuackData], Iterator[Row]]
            A dictionary (TypedDict) containing flattened data for a single item or if self.__raw is true, the
            unprocessed (Row) dictionary of row data

        References
        ----------
        https://censoredplanet.readthedocs.io/en/latest/hyperquackv1.html
        """
        for index, result in enumerate(scan.get('Results', [])):
            domain = self.__extract_domain_from_sent_field(result['Sent'])
            is_control = domain in QuackConstants.CONTROL_URLS.value
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
            try:
                if self.__anomalies and result['Success']:
                    # Not an anomaly.
                    continue
                received_fields = self.__parse_received_data(received)
                # Calculate censorship if required
                matches_blockpage = 0
                if self.__compare and result['Success'] and not scan['Blocked']:
                    matches_blockpage = -1
                elif self.__compare and len(received_fields['received_body']) > 0:
                    matches_blockpage = self.__blockpage_match(received_fields['received_body'])
                # If we only want labeled data and censorship is still undetermined, continue to the next row.
                if matches_blockpage == 0 and self.__labeled:
                    continue
            except KeyError:
                # There's something out of spec with this item.
                continue

            # Process time strings into a unix timestamp.
            start = isoparse(result['StartTime'])
            end = isoparse(result['EndTime'])

            # Create row data.
            row = Row(
                ip=scan['Server'],
                location= '',
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
            if self.__raw:
                result = row
            else:
                result = self.__process_row(row)
            yield result

    def __process_hyperquack_v2(self, scan: Dict) -> Union[Iterator[TokenizedQuackData], Iterator[Row]]:
        """
        Process a line of Echo/Discard/HTTP/S data in HyperQuack V2 format.

        Parameters
        ----------
        scan: dict
            A loaded json object containing the parsed content of the line

        Yields
        -------
        Union[Iterator[TokenizedQuackData], Iterator[Row]]
            A dictionary (TypedDict) containing flattened data for a single item or if self.__raw is true, the
            unprocessed (Row) dictionary of row data

        References
        ----------
        https://censoredplanet.readthedocs.io/en/latest/hyperquackv2.html

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
            try:
                if self.__anomalies and response['matches_template']:
                    # Not an anomaly.
                    continue
                received_fields = self.__parse_received_data(received)
                matches_blockpage = 0
                # Calculate censorship if required
                if self.__compare:
                    if response['matches_template'] and not scan['anomaly']:
                        matches_blockpage = -1
                    elif len(received_fields['received_body']) > 0:
                        matches_blockpage = self.__blockpage_match(received_fields['received_body'])
                    # If we only want labeled data and censorship is still undetermined, continue to the next row.
                    if matches_blockpage == 0 and self.__labeled:
                        continue
            except KeyError:
                # There's something out of spec with this item.
                continue

            # Process time strings into a unix timestamp.
            start = isoparse(response['start_time'])
            end = isoparse(response['end_time'])

            # Create row data.
            row = Row(
                ip=scan['vp'],
                location=scan['location']['country_name'],
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
            if self.__raw:
                result = row
            else:
                result = self.__process_row(row)
            yield result

    def __process_row(self, row: Row) -> TokenizedQuackData:
        """
        Transforms flattened data in a Row into a torch.Tensor with metadata.

        Parameters
        ----------
        row: Row

        Returns
        -------
        TokenizedQuackData
        """
        if not len(row['location']):
            try:
                lookup = self.__ip2geo.country(row['ip'])
                country = lookup.country.name
            except geoip2.errors.AddressNotFoundError:
                country = None
        else:
            country = row['location']

        metadata = {
            'domain': row['domain'],
            'ip': row['ip'],
            'location': country,
            'timestamp': row['start_time'],
            'censored': row['censored']
        }
        # Row keys with static length data.
        static_keys = ('success', 'anomaly', 'controls_failed', 'stateful_block', 'start_time', 'end_time', 'received_tls_version', 'received_tls_cipher_suite', 'received_tls_cert')
        # Row keys with variable length (text) data.
        # #> Skipping 'received_tls_cert' for now.
        text_keys = ('sent', 'received_status', 'received_headers', 'received_body')
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
        encoded = self.__xlmr.encode(concatenated).numpy()  # Type: np.ndarray
        # We keep a map of xlmr tokens actually used to reduce the scale of our models.
        mapped = np.zeros(encoded.shape, dtype=encoded.dtype)
        for index, value in enumerate(encoded):
            try:
                token = self.__vocab[value]
            except KeyError:
                token = self.__vocab_next
                self.__vocab[value] = self.__vocab_next
                self.__vocab_next += 1
            mapped[index] = token
        meta_tensor = TokenizedQuackData(
            metadata=metadata,
            static_size=np.array(static_dimension),
            variable_text=mapped
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

        match = re.search(QuackConstants.SENT_PATTERN.value, sent)
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
        Processes data found in the "received" key of a row.

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
            match = re.search(r'^HTTP/[\d.]+\s(\d+)', received)
            if match:
                data['received_status'] = match.group(1)
            data['received_body'] = received
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
        Uses the regular expression matcher provide by Censored Planet.

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

