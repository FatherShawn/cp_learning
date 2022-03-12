# cp_flatten module

CensoredPlanetFlatten and helper classes to flatten raw Censored Planet data.


### _class_ cp_flatten.QuackConstants(value)
Bases: `enum.Enum`

An Enum to contain constants used in this project.


#### SENT_PATTERN(_: st_ _ = 'GET (.\*) HTTP/1.1\\r\\nHost: (.\*)\\r\\n_ )

#### CONTROL_URLS(_: List[str_ _ = ['example5718349450314.com', 'rtyutgyhefdafioasfjhjhi.com'_ )

#### VOCAB(_ = 681_ )

#### XLMR_PAD(_: in_ _ = _ )

#### TIME_FLOOR(_: floa_ _ = 1625112000._ )

#### TIME_CEILING(_: floa_ _ = 1656648000._ )

#### \__module__(_ = 'cp_flatten_ )

### _class_ cp_flatten.Row(\*args, \*\*kwargs)
Bases: `dict`

A data structure for a single flattened row of CP data.

If data is labeled:
censorship: 1 => definitely, 0 => unknown, -1 => definitely not

If data is unlabeled, censorship defaults to 0.


#### ip(_: st_ )

#### location(_: st_ )

#### domain(_: st_ )

#### anomaly(_: boo_ )

#### controls_failed(_: boo_ )

#### stateful_block(_: boo_ )

#### success(_: boo_ )

#### error(_: st_ )

#### start_time(_: floa_ )

#### end_time(_: floa_ )

#### censored(_: in_ )

#### received_tls_version(_: in_ )

#### received_tls_cipher_suite(_: in_ )

#### received_tls_cert(_: st_ )

#### sent(_: st_ )

#### received_status(_: st_ )

#### received_headers(_: st_ )

#### received_body(_: st_ )

#### \__annotations__(_ = {'anomaly': <class 'bool'>, 'censored': <class 'int'>, 'controls_failed': <class 'bool'>, 'domain': <class 'str'>, 'end_time': <class 'float'>, 'error': <class 'str'>, 'ip': <class 'str'>, 'location': <class 'str'>, 'received_body': <class 'str'>, 'received_headers': <class 'str'>, 'received_status': <class 'str'>, 'received_tls_cert': <class 'str'>, 'received_tls_cipher_suite': <class 'int'>, 'received_tls_version': <class 'int'>, 'sent': <class 'str'>, 'start_time': <class 'float'>, 'stateful_block': <class 'bool'>, 'success': <class 'bool'>_ )

#### \__dict__(_ = mappingproxy({'__module__': 'cp_flatten', '__annotations__': {'ip': <class 'str'>, 'location': <class 'str'>, 'domain': <class 'str'>, 'anomaly': <class 'bool'>, 'controls_failed': <class 'bool'>, 'stateful_block': <class 'bool'>, 'success': <class 'bool'>, 'error': <class 'str'>, 'start_time': <class 'float'>, 'end_time': <class 'float'>, 'censored': <class 'int'>, 'received_tls_version': <class 'int'>, 'received_tls_cipher_suite': <class 'int'>, 'received_tls_cert': <class 'str'>, 'sent': <class 'str'>, 'received_status': <class 'str'>, 'received_headers': <class 'str'>, 'received_body': <class 'str'>}, '__doc__': '\\n    A data structure for a single flattened row of CP data.\\n\\n    If data is labeled:\\n    censorship: 1 => definitely, 0 => unknown, -1 => definitely not\\n\\n    If data is unlabeled, censorship defaults to 0.\\n    ', '__new__': <staticmethod object>, '__dict__': <attribute '__dict__' of 'Row' objects>, '__weakref__': <attribute '__weakref__' of 'Row' objects>, '__total__': True}_ )

#### \__module__(_ = 'cp_flatten_ )

#### _static_ \__new__(cls, /, \*args, \*\*kwargs)

#### \__total__(_ = Tru_ )

#### \__weakref__()
list of weak references to the object (if defined)


### _class_ cp_flatten.TokenizedQuackData(\*args, \*\*kwargs)
Bases: `dict`

A data structure to hold the flattened data.


#### metadata(_: dic_ )

#### static_size(_: numpy.ndarra_ )

#### variable_text(_: numpy.ndarra_ )

#### \__annotations__(_ = {'metadata': <class 'dict'>, 'static_size': <class 'numpy.ndarray'>, 'variable_text': <class 'numpy.ndarray'>_ )

#### \__dict__(_ = mappingproxy({'__module__': 'cp_flatten', '__annotations__': {'metadata': <class 'dict'>, 'static_size': <class 'numpy.ndarray'>, 'variable_text': <class 'numpy.ndarray'>}, '__doc__': '\\n    A data structure to hold the flattened data.\\n    ', '__new__': <staticmethod object>, '__dict__': <attribute '__dict__' of 'TokenizedQuackData' objects>, '__weakref__': <attribute '__weakref__' of 'TokenizedQuackData' objects>, '__total__': True}_ )

#### \__module__(_ = 'cp_flatten_ )

#### _static_ \__new__(cls, /, \*args, \*\*kwargs)

#### \__total__(_ = Tru_ )

#### \__weakref__()
list of weak references to the object (if defined)


### _class_ cp_flatten.CensoredPlanetFlatten(urls: Union[str, List[str]], vocab_path: str = '', compare: bool = False, labeled: bool = False, anomalies: bool = False, raw: bool = False)
Bases: `torch.utils.data.dataset.IterableDataset`, `webdataset.dataset.Shorthands`

Although (Webdataset)[[https://webdataset.github.io/webdataset/](https://webdataset.github.io/webdataset/)] may be able to handle all our pipeline needs,
my intention here is to take in the Censored Planet Quack data and pre-preprocess it into Pytorch Tensors.

The following are adapted from [https://github.com/censoredplanet/censoredplanet-analysis/blob/master/pipeline/metadata/flatten.py](https://github.com/censoredplanet/censoredplanet-analysis/blob/master/pipeline/metadata/flatten.py)


* process_hyperquack_v1


* process_hyperquack_v2


* extract_domain_from_sent_field


#### \__init__(urls: Union[str, List[str]], vocab_path: str = '', compare: bool = False, labeled: bool = False, anomalies: bool = False, raw: bool = False)

* **Parameters**

    
    * **urls** (*Union**[**str**, **List**[**str**]**]*) – Path or paths to pass to webdataset.dataset.ShardList. Points to Censored Planet .tar data files.


    * **vocab_path** (*str*) – Path to a .pyc file which holds a dictionary that maps an index sequence with tokens used from
    fairseq.models.roberta.model_xlmr.XLMRModel when flattening data.


    * **compare** (*bool*) – Should data be compared with Censored Planet blockpage signatures?


    * **labeled** (*bool*) – Should only data successfully precessed by blockpage matcher be returned?


    * **anomalies** (*bool*) – Should only data marked by Censored Planet as an anomaly be processed?


    * **raw** (*bool*) – Should the raw row be returned without processing into vectors?



#### \__getitem__(index)
Required by the parent of IterableDataset but not useful in this context, and not implemented by any of the
Webdataset implementations of IterableDataset.


#### \__iter__()
Iterates the data in the .tar files.


* **Returns**

    A dictionary (TypedDict) containing flattened data for a single item or if self.__raw is true, the
    unprocessed (Row) dictionary of row data is returned.



* **Return type**

    Union[Iterator[TokenizedQuackData], Iterator[Row]]



#### \__abstractmethods__(_ = frozenset({}_ )

#### \__module__(_ = 'cp_flatten_ )

#### \__type_class__(_ = Fals_ )

#### reinforce_type(expected_type)
Reinforce the type for DataPipe instance. And the ‘expected_type’ is required
to be a subtype of the original type hint to restrict the type requirement
of DataPipe instance.
