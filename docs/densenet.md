# densenet module

The densenet model with classes composed into the densenet class.


### _class_ densenet.QuackDenseNet(learning_rate: float = 0.1, learning_rate_min: float = 0.0001, lr_max_epochs: int = - 1, freeze: bool = True, \*args: Any, \*\*kwargs: Any)
Bases: `pytorch_lightning.core.lightning.LightningModule`

A modification of the Pytorch Densenet 121 pretrained model.

### References

[https://pytorch.org/hub/pytorch_vision_densenet/](https://pytorch.org/hub/pytorch_vision_densenet/)


#### configure_optimizers()
Configures the optimizer and learning rate scheduler objects.


* **Returns**

    A dictionary with keys:


    * optimizer: pt.optim.AdamW


    * lr_scheduler: pt.optim.lr_scheduler.CosineAnnealingLR




* **Return type**

    dict



#### forward(x: torch.Tensor)

* **Parameters**

    **x** (*pt.Tensor*) – The input, which should be (B, 3, H, W) shaped, where:
    B: batch size
    3: Densnet is trained on RGB = 3 channels
    H: height
    W: width



* **Returns**

    The output, which should be (B, 1) sized, of single probability floats.



* **Return type**

    pt.Tensor



#### predict_step(batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None)
Calls _common_step for step ‘predict’.


* **Parameters**

    
    * **batch** (*pt. Tuple**[**dict**, **pt.Tensor**]*) – An tuple of a metadata dictionary and the associated input data


    * **batch_idx** (*int*) – The index of the batch.  Required to match the parent signature.  Unused in our model.


    * **dataloader_idx** (*int*) – Index of the current dataloader.   Required to match the parent signature.  Unused in our model.



* **Returns**

    An tuple of the batch metadata dictionary and the associated output data



* **Return type**

    Tuple[dict, pt.Tensor]



#### test_epoch_end(outputs: List[Union[torch.Tensor, Dict[str, Any]]])
Called at the end of a test epoch with the output of all test steps.

Now that all the test steps are complete, we compute the metrics.


* **Parameters**

    **outputs** (*None*) – No outputs are passed on from test_step_end.



* **Return type**

    void



#### test_step(x: torch.Tensor, batch_index: int)
Calls _common_step for step ‘test’.


* **Parameters**

    
    * **x** (*pt. Tensor*) – An input tensor


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



#### test_step_end(outputs: dict, \*args, \*\*kwargs)
When using distributed backends, only a portion of the batch is inside the test_step.

    We calculate metrics here with the entire batch.


* **Parameters**

    
    * **outputs** (*dict*) – The return values from training_step for each batch part.


    * **args** (*Any*) – Matching to the parent constructor.


    * **kwargs** (*Any*) – Matching to the parent constructor.



* **Return type**

    void



#### training(_: boo_ )

#### training_step(x: torch.Tensor, batch_index: int)
Calls _common_step for step ‘train’.


* **Parameters**

    
    * **x** (*pt. Tensor*) – An input tensor


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



#### training_step_end(outputs: dict, \*args, \*\*kwargs)
When using distributed backends, only a portion of the batch is inside the training_step.
We calculate metrics here with the entire batch.


* **Parameters**

    
    * **outputs** (*dict*) – The return values from training_step for each batch part.


    * **args** (*Any*) – Matching to the parent constructor.


    * **kwargs** (*Any*) – Matching to the parent constructor.



* **Return type**

    void



#### validation_epoch_end(outputs: List[Union[torch.Tensor, Dict[str, Any]]])
Called at the end of a validation epoch with the output of all test steps.

Now that all the validation steps are complete, we compute the metrics.


* **Parameters**

    **outputs** (*None*) – No outputs are passed on from test_step_end.



* **Return type**

    void



#### validation_step(x: torch.Tensor, batch_index: int)
Calls _common_step for step ‘val’.


* **Parameters**

    
    * **x** (*pt. Tensor*) – An input tensor


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



#### validation_step_end(outputs: dict, \*args, \*\*kwargs)
When using distributed backends, only a portion of the batch is inside the validation_step.

    We calculate metrics here with the entire batch.


* **Parameters**

    
    * **outputs** (*dict*) – The return values from training_step for each batch part.


    * **args** (*Any*) – Matching to the parent constructor.


    * **kwargs** (*Any*) – Matching to the parent constructor.



* **Return type**

    void
