# cp_image_dataset module

Defines a class and data structures for flattened quack data stored as images and pixel tensors.


### _class_ cp_image_dataset.QuackImageData(\*args, \*\*kwargs)
Bases: `dict`

Flattened quack data and associated metadata.


#### metadata(_: dic_ )

#### pixels(_: numpy.ndarra_ )

### _class_ cp_image_dataset.QuackImageDataset(path: str)
Bases: `torch.utils.data.dataset.Dataset`

Iterates or selectively retrieves items from a collection of python pickle files which contain QuackImageData

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
Metadata for the response is stored in the metadata key of the QuackImageData typed dictionary:

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

Each QuackImageData stores one numpy array:

pixels

    A (224, 224) numpy array of pixel data


#### censored()
Getter for the value of self.__censored.


* **Return type**

    int



#### uncensored()
Getter for the value of self.__uncensored.


* **Return type**

    int



#### undetermined()
Getter for the value of self.__undetermined.


* **Return type**

    int
