# cp_dataset module


### _class_ cp_dataset.QuackIterableDataset(path: str)
Bases: `torch.utils.data.dataset.Dataset`

Iterates or selectively retrieves items from a collection of python pickle files which contain TokenizedQuackData

Metadata stored in metadata.pyc:

length

    The number of responses in the file.

censored

    The number of responses labeled ‘censored’ by existing Censored Planet process. Dataset must have been
    flattened as “Labeled”

undetermined

    The number of unlabeled responses.

uncensored

    The number of responses labeled ‘censored’ by existing Censored Planet process. Dataset must have been
    flattened as “Labeled”

Each response is stored in a single .pyc file, named with the index number of the response, zero based.
Metadata for the response is stored in the metadata key of the TokenizedQuackData typed dictionary:

domain

    The domain under test

ip

    The IPv4 address for this test

location

    The country returned by MMDB for the IP address

timestamp

    A Unix timestamp for the time of the test

censored

    1 if censored, -1 if uncensored, 0 as default (undetermined)

Each TokenizedQuackData stores two numpy arrays:

static_size

    Data that is a fixed size.  See cp_flatten.CensoredPlanetFlatten.__process_row

variable_text

    Text data that has been encoded (tokenized) using the XLMR pretrained model.


#### \__init__(path: str)
Constructs the QuackIterableDataset object.


* **Parameters**

    **path** (*str*) – A path to the top level of the data directories.



#### \__iter__()
Iterates through all data points in the dataset.


* **Return type**

    Iterator[[TokenizedQuackData](cp_flatten.md#cp_flatten.TokenizedQuackData)]



#### \__getitem__(index: int)
> Implements a required method to access a single data point by index.


* **Parameters**

    **index** (*int*) – The index of the data item.



* **Returns**

    A dictionary (TypedDict) containing the data.



* **Return type**

    [TokenizedQuackData](cp_flatten.md#cp_flatten.TokenizedQuackData)



#### \__len__()

* **Returns**

    The length of this dataset.



* **Return type**

    int



#### \__module__(_ = 'cp_dataset_ )

#### \__parameters__(_ = (_ )

#### censored()
Getter for the value of self.__censored.


* **Return type**

    int



#### undetermined()
Getter for the value of self.__undetermined.


* **Return type**

    int



#### uncensored()
Getter for the value of self.__uncensored.


* **Return type**

    int



#### data_width()
Getter for the value of self.__max_width.


* **Return type**

    int
