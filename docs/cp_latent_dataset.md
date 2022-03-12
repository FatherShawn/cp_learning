# cp_latent_dataset module

Defines a class and data structures for quack data en tensors.


### _class_ cp_latent_dataset.QuackLatentData(\*args, \*\*kwargs)
Bases: `dict`

Encoded quack data and associated metadata.


#### metadata(_: dic_ )

#### encoded(_: numpy.ndarra_ )

#### \__annotations__(_ = {'encoded': <class 'numpy.ndarray'>, 'metadata': <class 'dict'>_ )

#### \__dict__(_ = mappingproxy({'__module__': 'cp_latent_dataset', '__annotations__': {'metadata': <class 'dict'>, 'encoded': <class 'numpy.ndarray'>}, '__doc__': '\\n    Encoded quack data and associated metadata.\\n    ', '__new__': <staticmethod object>, '__dict__': <attribute '__dict__' of 'QuackLatentData' objects>, '__weakref__': <attribute '__weakref__' of 'QuackLatentData' objects>, '__total__': True}_ )

#### \__module__(_ = 'cp_latent_dataset_ )

#### _static_ \__new__(cls, /, \*args, \*\*kwargs)

#### \__total__(_ = Tru_ )

#### \__weakref__()
list of weak references to the object (if defined)


### _class_ cp_latent_dataset.QuackLatentDataset(path: str)
Bases: `torch.utils.data.dataset.Dataset`

Iterates or selectively retrieves items from a collection of python pickle files which contain QuackLatentData

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

Each QuackLatentData stores one numpy array:

pixels

    A (224, 224) numpy array of pixel data


#### \__init__(path: str)
Constructs QuackImageDataset.


* **Parameters**

    **paths** (*str*) – A path to the top level of the data directories.



#### \__iter__()
Iterates through all data points in the dataset.


* **Return type**

    [QuackImageData](cp_image_dataset.md#cp_image_dataset.QuackImageData)



#### \__getitem__(index)
Implements a required method to access a single data point by index.


* **Parameters**

    **index** (*int*) – The index of the data item.



* **Return type**

    [QuackImageData](cp_image_dataset.md#cp_image_dataset.QuackImageData)



#### \__len__()

#### \__module__(_ = 'cp_latent_dataset_ )

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
