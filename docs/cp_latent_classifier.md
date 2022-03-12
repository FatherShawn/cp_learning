# cp_latent_classifier module

The latent classifier model with classes composed into the classifier class.


### _class_ cp_latent_classifier.QuackLatentClassifier(initial_size: int, learning_rate: float = 0.1, learning_rate_min: float = 0.0001, lr_max_epochs: int = - 1, \*args: Any, \*\*kwargs: Any)
Bases: `pytorch_lightning.core.lightning.LightningModule`

A binary classifier which operates on encoded tensors produced by
autoencoder.QuackAutoEncoder.


#### \__init__(initial_size: int, learning_rate: float = 0.1, learning_rate_min: float = 0.0001, lr_max_epochs: int = - 1, \*args: Any, \*\*kwargs: Any)
Constructor for QuackLatentClassifier.


* **Parameters**

    
    * **learning_rate** (*float*) – Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR


    * **learning_rate_min** (*float*) – Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR


    * **lr_max_epochs** (*int*) – Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR


    * **args** (*Any*) – Passed to the parent constructor.


    * **kwargs** (*Any*) – Passed to the parent constructor.



#### forward(x: torch.Tensor)
Process a batch of input through the model.


* **Parameters**

    **x** (*pt.Tensor*) – The input



* **Returns**

    The output, which should be (B, 1) sized, of single probability floats.



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



#### \__module__(_ = 'cp_latent_classifier_ )

#### training(_: boo_ )
