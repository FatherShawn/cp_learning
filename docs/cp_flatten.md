# cp_flatten module

CensoredPlanetFlatten and helper classes to flatten raw Censored Planet data.


### _class_ cp_flatten.CensoredPlanetFlatten(urls: Union[str, List[str]], vocab_path: str, compare: bool = False, labeled: bool = False, anomalies: bool = False)
Bases: `torch.utils.data.dataset.IterableDataset`, `webdataset.dataset.Shorthands`

Although (Webdataset)[[https://webdataset.github.io/webdataset/](https://webdataset.github.io/webdataset/)] may be able to handle all our pipeline needs,
my intention here is to take in the Censored Planet Quack data and pre-preprocess it into Pytorch Tensors.

The following are adapted from [https://github.com/censoredplanet/censoredplanet-analysis/blob/master/pipeline/metadata/flatten.py](https://github.com/censoredplanet/censoredplanet-analysis/blob/master/pipeline/metadata/flatten.py)


* process_hyperquack_v1


* process_hyperquack_v2


* extract_domain_from_sent_field


* **Parameters**

    
    * **self.__shards** (*ShardList*) – The dataset to use a pipeline source.


    * **self.__blockpage_matcher** ([*BlockpageMatcher*](blockpage.md#blockpage.BlockpageMatcher)) – The blockpage matching utility.


    * **self.__labeled** (*bool*) – Should the data be labeled as censored?


    * **self.__xlmr** (*object*) – The XLMR pretrained tokenizer.



#### reinforce_type(expected_type)
Reinforce the type for DataPipe instance. And the ‘expected_type’ is required
to be a subtype of the original type hint to restrict the type requirement
of DataPipe instance.


### _class_ cp_flatten.QuackConstants(value)
Bases: `enum.Enum`

An Enum to contain constants used in this project.


#### CONTROL_URLS(_: List[str_ _ = ['example5718349450314.com', 'rtyutgyhefdafioasfjhjhi.com'_ )

#### SENT_PATTERN(_: st_ _ = 'GET (.\*) HTTP/1.1\\r\\nHost: (.\*)\\r\\n_ )

#### TIME_CEILING(_: floa_ _ = 1656648000._ )

#### TIME_FLOOR(_: floa_ _ = 1625112000._ )

#### VOCAB(_ = 681_ )

#### XLMR_PAD(_: in_ _ = _ )

### _class_ cp_flatten.Row(\*args, \*\*kwargs)
Bases: `dict`

A data structure for a single flattened row of CP data.

If data is labeled:
censorship: 1 => definitely, 0 => unknown, -1 => definitely not

If data is unlabeled, censorship defaults to 0.


#### anomaly(_: boo_ )

#### censored(_: in_ )

#### controls_failed(_: boo_ )

#### domain(_: st_ )

#### end_time(_: floa_ )

#### error(_: st_ )

#### ip(_: st_ )

#### location(_: st_ )

#### received_body(_: st_ )

#### received_headers(_: st_ )

#### received_status(_: st_ )

#### received_tls_cert(_: st_ )

#### received_tls_cipher_suite(_: in_ )

#### received_tls_version(_: in_ )

#### sent(_: st_ )

#### start_time(_: floa_ )

#### stateful_block(_: boo_ )

#### success(_: boo_ )

### _class_ cp_flatten.TokenizedQuackData(\*args, \*\*kwargs)
Bases: `dict`

A data structure to hold the flattened data.


#### metadata(_: dic_ )

#### static_size(_: numpy.ndarra_ )

#### variable_text(_: numpy.ndarra_ )
