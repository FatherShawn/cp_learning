# cp_tokenized_data module

Extends pytorch_lightning.core.datamodule.LightningDataModule and wraps QuackIterableDataset for use by
pytorch_lightning.trainer.trainer.Trainer


### cp_tokenized_data.pad_right(batch: List[dict])
Receives a list of Tensors with B elements.  Calculates the widest tensor, which is length T. Pads all
narrower tensors to T with zeros.  Returns a (B x T) shaped tensor.


* **Parameters**

    **batch** (*List**[**pt.Tensor**]*) – A list of tensors in the batch.



* **Return type**

    pt.Tensor



### cp_tokenized_data.pad_right_with_meta(batch: List[dict])
Receives a list of TokenizedQuackData with B elements.  Calculates the widest tensor, which is length T. Pads all
narrower tensors to T with zeros.  Returns a (B x T) shaped tensor.


* **Parameters**

    **batch** (*List**[*[*TokenizedQuackData*](cp_flatten.md#cp_flatten.TokenizedQuackData)*]*) – A list of TokenizedQuackData (TypedDict) in the batch.



* **Returns**

    A tuple of a list of metadata and a batch tensor.



* **Return type**

    Tuple[List[dict], pt.Tensor]



### cp_tokenized_data.concatenate_data(item: dict)
Concatenates the static and text data into a single numpy array.


* **Parameters**

    **item** (*dict*) – A TypedDict cp_flatten.TokenizedQuackData



* **Returns**

    The concatenated data.



* **Return type**

    np.ndarray



### _class_ cp_tokenized_data.QuackTokenizedDataModule(\*args: Any, \*\*kwargs: Any)
Bases: `pytorch_lightning.core.datamodule.LightningDataModule`


#### \__init__(data_dir: str, batch_size: int = 64, workers: int = 0, train_transforms=None, val_transforms=None, test_transforms=None, dims=None)
Constructs QuackTokenizedDataModule.


* **Parameters**

    
    * **data_dir** (*str*) – The path to top dir of the QuackIterableDataset.


    * **batch_size** (*int*) – The batch size to pass to the torch.utils.data.dataloader.DataLoader


    * **workers** (*int*) – The number of workers to pass to the torch.utils.data.dataloader.DataLoader


    * **train_transforms** – deprecated: DataModule property train_transforms was deprecated in
    pytorch_lightning.core.datamodule.LightningDataModule v1.5 and will be removed in v1.7.


    * **val_transforms** – deprecated: DataModule property val_transforms was deprecated in
    pytorch_lightning.core.datamodule.LightningDataModule v1.5 and will be removed in v1.7.


    * **test_transforms** – deprecated: DataModule property test_transforms was deprecated in
    pytorch_lightning.core.datamodule.LightningDataModule v1.5 and will be removed in v1.7.


    * **dims** – deprecated: DataModule property dims was deprecated in
    pytorch_lightning.core.datamodule.LightningDataModule v1.5 and will be removed in v1.7.



#### train_dataloader()
Constructs and returns the training dataloader using collate function pad_right.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### test_dataloader()
Constructs and returns the testing dataloader using collate function pad_right.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### val_dataloader()
Constructs and returns the validation dataloader using collate function pad_right.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### predict_dataloader()
Constructs and returns the inference dataloader using collate function pad_right_with_meta.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### get_width()
Returns data_width() from the cp_dataset.QuackIterableDataset loaded in this data module.


* **Return type**

    int



#### \__module__(_ = 'cp_tokenized_data_ )
