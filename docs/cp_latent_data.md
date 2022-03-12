# cp_latent_data module


### _class_ cp_latent_data.QuackLatentCollator(step: str)
Bases: `object`


#### \__init__(step: str)

#### \__call__(batch: List[dict], \*args, \*\*kwargs)
Call self as a function.


#### \__dict__(_ = mappingproxy({'__module__': 'cp_latent_data', '__init__': <function QuackLatentCollator.__init__>, '__call__': <function QuackLatentCollator.__call__>, '_QuackLatentCollator__collate_labels': <function QuackLatentCollator.__collate_labels>, '_QuackLatentCollator__collate_predict': <function QuackLatentCollator.__collate_predict>, '__dict__': <attribute '__dict__' of 'QuackLatentCollator' objects>, '__weakref__': <attribute '__weakref__' of 'QuackLatentCollator' objects>, '__doc__': None, '__annotations__': {}}_ )

#### \__module__(_ = 'cp_latent_data_ )

#### \__weakref__()
list of weak references to the object (if defined)


### _class_ cp_latent_data.QuackLatentDataModule(\*args: Any, \*\*kwargs: Any)
Bases: `pytorch_lightning.core.datamodule.LightningDataModule`


#### \__init__(data_dir: str, batch_size: int = 64, workers: int = 0)

#### train_dataloader()
Constructs and returns the train dataloader using an `QuackLatentCollator` object configured for training.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### test_dataloader()
Constructs and returns the test dataloader using an `QuackLatentCollator` object configured for testing.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### val_dataloader()
Constructs and returns the validation dataloader using an `QuackLatentCollator` object
configured for validation.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### predict_dataloader()
Constructs and returns the prediction dataloader using an `QuackLatentCollator` object configured
for prediction.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### \__module__(_ = 'cp_latent_data_ )
