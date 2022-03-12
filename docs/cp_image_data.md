# cp_image_data module


### _class_ cp_image_data.QuackImageTransformer(step: str, strategy: str)
Bases: `object`


#### \__init__(step: str, strategy: str)

#### \__call__(batch: List[dict], \*args, \*\*kwargs)
Call self as a function.


#### \__dict__(_ = mappingproxy({'__module__': 'cp_image_data', '__init__': <function QuackImageTransformer.__init__>, '__call__': <function QuackImageTransformer.__call__>, '_QuackImageTransformer__collate_labels': <function QuackImageTransformer.__collate_labels>, '_QuackImageTransformer__collate_predict': <function QuackImageTransformer.__collate_predict>, '__dict__': <attribute '__dict__' of 'QuackImageTransformer' objects>, '__weakref__': <attribute '__weakref__' of 'QuackImageTransformer' objects>, '__doc__': None, '__annotations__': {}}_ )

#### \__module__(_ = 'cp_image_data_ )

#### \__weakref__()
list of weak references to the object (if defined)


### _class_ cp_image_data.QuackImageDataModule(\*args: Any, \*\*kwargs: Any)
Bases: `pytorch_lightning.core.datamodule.LightningDataModule`


#### \__init__(data_dir: str, batch_size: int = 64, workers: int = 0, simple_transforms: bool = True)

#### train_dataloader()
Constructs and returns the training dataloader using an QuackImageTransformer object configured for training.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### test_dataloader()
Constructs and returns the testing dataloader using an QuackImageTransformer object configured for testing.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### val_dataloader()
Constructs and returns the validation dataloader using
an QuackImageTransformer object configured for validation.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### \__module__(_ = 'cp_image_data_ )

#### predict_dataloader()
Constructs and returns the prediction dataloader using
an QuackImageTransformer object configured for prediction.


* **Return type**

    torch.utils.data.dataloader.DataLoader
