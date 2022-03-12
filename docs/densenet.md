# densenet module

The densenet model with classes composed into the densenet class.


### _class_ densenet.CensoredDataWriter(write_interval: str = 'batch', storage_path: str = '~/data')
Bases: `pytorch_lightning.callbacks.prediction_writer.BasePredictionWriter`

Extends pytorch_lightning.callbacks.prediction_writer.BasePredictionWriter
to store metadata for detected censorship.


#### \__init__(write_interval: str = 'batch', storage_path: str = '~/data')
Constructor for AutoencoderWriter.


* **Parameters**

    
    * **write_interval** (*str*) – See parent class BasePredictionWriter


    * **storage_path** (*str*) – A string file path to the directory in which output will be stored.



#### write_on_batch_end(trainer: pytorch_lightning.trainer.trainer.Trainer, pl_module: pytorch_lightning.core.lightning.LightningModule, prediction: Any, batch_indices: Optional[Sequence[int]], batch: Any, batch_idx: int, dataloader_idx: int)
Logic to write the results of a single batch to files.


* **Parameters**

    **class.** (*Parameter signature defined in the parent*) – 



* **Return type**

    void



#### write_on_epoch_end(trainer: pytorch_lightning.trainer.trainer.Trainer, pl_module: pytorch_lightning.core.lightning.LightningModule, predictions: Sequence[Any], batch_indices: Optional[Sequence[Any]])
Implementation expected by the base class.  Unused in our case.


#### \__abstractmethods__(_ = frozenset({}_ )

#### \__module__(_ = 'densenet_ )

### _class_ densenet.QuackDenseNet(learning_rate: float = 0.1, learning_rate_min: float = 0.0001, lr_max_epochs: int = - 1, freeze: bool = True, \*args: Any, \*\*kwargs: Any)
Bases: `pytorch_lightning.core.lightning.LightningModule`

A modification of the Pytorch Densenet 121 pretrained model.

### References

[https://pytorch.org/hub/pytorch_vision_densenet/](https://pytorch.org/hub/pytorch_vision_densenet/)


#### \__init__(learning_rate: float = 0.1, learning_rate_min: float = 0.0001, lr_max_epochs: int = - 1, freeze: bool = True, \*args: Any, \*\*kwargs: Any)
Constructor for QuackDenseNet.


* **Parameters**

    
    * **learning_rate** (*float*) – Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR


    * **learning_rate_min** (*float*) – Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR


    * **lr_max_epochs** (*int*) – Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR


    * **freeze** (*bool*) – Should the image analyzing layers of the pre-trained Densenet be frozen?


    * **args** (*Any*) – Passed to the parent constructor.


    * **kwargs** (*Any*) – Passed to the parent constructor.



#### set_balanced_loss(balance: float)
Using the balance factor passed, even the loss.


* **Parameters**

    **balance** (*float*) – The proportion of negative samples / positive samples.



* **Return type**

    void


### References

[https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bceloss#bcewithlogitsloss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bceloss#bcewithlogitsloss)


#### forward(x: torch.Tensor)
Process a batch of input through the model.


* **Parameters**

    **x** (*pt.Tensor*) – The input, which should be (B, 3, H, W) shaped, where:
    B: batch size
    3: Densnet is trained on RGB = 3 channels
    H: height
    W: width



* **Returns**

    The output, which should be (B, 1) sized, of confidence score floats.



* **Return type**

    pt.Tensor



#### training_step(x: Tuple[torch.Tensor, torch.Tensor], batch_index: int)
Calls _common_step for step ‘train’.


* **Parameters**

    
    * **x** (*Tuple**[**pt.Tensor**, **pt.Tensor**]*) – The input tensor and a label tensor


    * **batch_index** (*int*) – The index of the batch.  Required to match the parent signature.  Unused in our model.



* **Returns**

    Format expected by the parent class. Has three keys:

    loss

        The loss returned by _common_step.

    expected

        The labels from the batch returned by _common_step.

    predicted

        The predicted labels from the batch returned by _common_step.




* **Return type**

    dict



#### validation_step(x: Tuple[torch.Tensor, torch.Tensor], batch_index: int)
Calls _common_step for step ‘val’.


* **Parameters**

    
    * **x** (*Tuple**[**pt.Tensor**, **pt.Tensor**]*) – The input tensor and a label tensor


    * **batch_index** (*int*) – The index of the batch.  Required to match the parent signature.  Unused in our model.



* **Returns**

    Format expected by the parent class. Has three keys:

    loss

        The loss returned by _common_step.

    expected

        The labels from the batch returned by _common_step.

    predicted

        The predicted labels from the batch returned by _common_step.




* **Return type**

    dict



#### test_step(x: Tuple[torch.Tensor, torch.Tensor], batch_index: int)
Calls _common_step for step ‘test’.


* **Parameters**

    
    * **x** (*Tuple**[**pt.Tensor**, **pt.Tensor**]*) – The input tensor and a label tensor


    * **batch_index** (*int*) – The index of the batch.  Required to match the parent signature.  Unused in our model.



* **Returns**

    Format expected by the parent class. Has three keys:

    loss

        The loss returned by _common_step.

    expected

        The labels from the batch returned by _common_step.

    predicted

        The predicted labels from the batch returned by _common_step.




* **Return type**

    dict



#### predict_step(batch: Tuple[torch.Tensor, List[dict]], batch_idx: int, dataloader_idx: Optional[int] = None)
Calls forward for prediction.


* **Parameters**

    
    * **batch** (*Tuple**[**pt.Tensor**, **List**[**dict**]**]*) – An tuple of a metadata dictionary and the associated input data


    * **batch_idx** (*int*) – The index of the batch.  Required to match the parent signature.  Unused in our model.


    * **dataloader_idx** (*int*) – Index of the current dataloader.   Required to match the parent signature.  Unused in our model.



* **Returns**

    An tuple of the batch metadata dictionary and the associated output data



* **Return type**

    Tuple[dict, pt.Tensor]



#### training_step_end(outputs: dict, \*args, \*\*kwargs)
When using distributed backends, only a portion of the batch is inside the training_step.
We calculate metrics here with the entire batch.


* **Parameters**

    
    * **outputs** (*dict*) – The return values from training_step for each batch part.


    * **args** (*Any*) – Matching to the parent constructor.


    * **kwargs** (*Any*) – Matching to the parent constructor.



* **Return type**

    void



#### validation_step_end(outputs: dict, \*args, \*\*kwargs)
When using distributed backends, only a portion of the batch is inside the validation_step.
We calculate metrics here with the entire batch.


* **Parameters**

    
    * **outputs** (*dict*) – The return values from training_step for each batch part.


    * **args** (*Any*) – Matching to the parent constructor.


    * **kwargs** (*Any*) – Matching to the parent constructor.



* **Return type**

    void



#### test_step_end(outputs: dict, \*args, \*\*kwargs)
When using distributed backends, only a portion of the batch is inside the test_step.
We calculate metrics here with the entire batch.


* **Parameters**

    
    * **outputs** (*dict*) – The return values from training_step for each batch part.


    * **args** (*Any*) – Matching to the parent constructor.


    * **kwargs** (*Any*) – Matching to the parent constructor.



* **Return type**

    void



#### test_epoch_end(outputs: List[Union[torch.Tensor, Dict[str, Any]]])
Called at the end of a test epoch with the output of all test steps.

Now that all the test steps are complete, we compute the metrics.


* **Parameters**

    **outputs** (*None*) – No outputs are passed on from test_step_end.



* **Return type**

    void



#### validation_epoch_end(outputs: List[Union[torch.Tensor, Dict[str, Any]]])
Called at the end of a validation epoch with the output of all test steps.

Now that all the validation steps are complete, we compute the metrics.


* **Parameters**

    **outputs** (*None*) – No outputs are passed on from test_step_end.



* **Return type**

    void



#### configure_optimizers()
Configures the optimizer and learning rate scheduler objects.


* **Returns**

    A dictionary with keys:


    * optimizer: pt.optim.AdamW


    * lr_scheduler: pt.optim.lr_scheduler.CosineAnnealingLR




* **Return type**

    dict



#### \__module__(_ = 'densenet_ )

#### training(_: boo_ )
