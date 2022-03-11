# cp_image_data module


### _class_ cp_image_data.QuackImageDataModule(\*args: Any, \*\*kwargs: Any)
Bases: `pytorch_lightning.core.datamodule.LightningDataModule`


#### predict_dataloader()
Constructs and returns the prediction dataloader using
an QuackImageTransformer object configured for prediction.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### test_dataloader()
Constructs and returns the testing dataloader using an QuackImageTransformer object configured for testing.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### train_dataloader()
Constructs and returns the training dataloader using an QuackImageTransformer object configured for training.


* **Return type**

    torch.utils.data.dataloader.DataLoader



#### val_dataloader()
Constructs and returns the validation dataloader using
an QuackImageTransformer object configured for validation.


* **Return type**

    torch.utils.data.dataloader.DataLoader



### _class_ cp_image_data.QuackImageTransformer(step: str, strategy: str)
Bases: `object`
