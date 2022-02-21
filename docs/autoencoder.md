# autoencoder module

The autoencoder class along with classes that compose the autoencoder class and helper functions.


### _class_ autoencoder.AttentionModule()
Bases: `torch.nn.modules.module.Module`

Applies attention to the hidden states.

### References

Edward Raff. 2021. Inside Deep Learning: Math, Algorithms, Models. Manning
Publications Co., Shelter Island, New York


#### forward(states: torch.Tensor, attention_scores, mask: Union[None, torch.Tensor] = None)
Processes the attention inputs.


* **Parameters**

    
    * **states** (*pt.Tensor*) – (B, T, H) shape giving the T different possible inputs attention_scores:
    (B, T, 1) score for each item at each context


    * **mask** (*Union**[**None**, **pt.Tensor**]*) – None if all items are present. Else a boolean tensor of shape
    (B, T), with True indicating which items are present / valid.



* **Returns**

    A tuple with two tensors. The first tensor is the final context from applying the attention to the
    states (B, H) shape. The second tensor is the weights for each state with shape (B, T, 1).



* **Return type**

    Tuple[pt.Tensor, pt.Tensor]



#### training(_: boo_ )

### _class_ autoencoder.AttentionScore(dim: int)
Bases: `torch.nn.modules.module.Module`

Uses the dot-product to calculate the attention scores.

### References

Edward Raff. 2021. Inside Deep Learning: Math, Algorithms, Models. Manning
Publications Co., Shelter Island, New York


#### forward(states: torch.Tensor, context: torch.Tensor)
> Computes the dot-product score:

> :math\`score(h_t, c) =

rac{h^T_t cdot c}{sqrt{H}}\`

    with values of h taken from the states parameter, c from the context paramter and H is the dim parameter
    passed at construction.

    states: pt.Tensor

        Hidden states; shape (B, T, H)

    context: pt.Tensor

        Context values; shape (B, H)

    pt.Tensor

        Scores for T items, based on context; shape (B, T, 1)


#### training(_: boo_ )

### _class_ autoencoder.AutoencoderWriter(write_interval: str = 'batch', storage_path: str = '~/data', filtered: bool = False, evaluate: bool = False)
Bases: `pytorch_lightning.callbacks.prediction_writer.BasePredictionWriter`

Extends prediction writer to store encoded Quack data.


#### write_on_batch_end(trainer: pytorch_lightning.trainer.trainer.Trainer, pl_module: pytorch_lightning.core.lightning.LightningModule, prediction: Any, batch_indices: Optional[Sequence[int]], batch: Any, batch_idx: int, dataloader_idx: int)
Logic to write the results of a single batch to files.


* **Parameters**

    **class.** (*Parameter signature defined in the parent*) – 



* **Return type**

    void



#### write_on_epoch_end(trainer: pytorch_lightning.trainer.trainer.Trainer, pl_module: pytorch_lightning.core.lightning.LightningModule, predictions: Sequence[Any], batch_indices: Optional[Sequence[Any]])
Logic to write the metadata for the data processed to file.


* **Parameters**

    **class.** (*Parameter signature defined in the parent*) – 



* **Return type**

    void



### _class_ autoencoder.QuackAutoEncoder(num_embeddings: int, embed_size: int, hidden_size: int, layers: int = 1, max_decode_length: Optional[int] = None, learning_rate: float = 0.1, learning_rate_min: float = 0.0001, lr_max_epochs: int = - 1, \*args: Any, \*\*kwargs: Any)
Bases: `pytorch_lightning.core.lightning.LightningModule`

A Sequence-to-Sequence based autoencoder

### References

Edward Raff. 2021. Inside Deep Learning: Math, Algorithms, Models. Manning
Publications Co., Shelter Island, New York


#### configure_optimizers()
Configures the optimizer and learning rate scheduler objects.


* **Returns**

    A dictionary with keys:


    * optimizer: pt.optim.AdamW


    * lr_scheduler: pt.optim.lr_scheduler.CosineAnnealingLR




* **Return type**

    dict



#### forward(x: torch.Tensor)
We put just the encoding process in forward.  The twin decoding process will only be found in
the common step used in training and testing. This prepares the model for its intended use as
encoding and condensing latent features.


* **Parameters**

    **x** (*pt.Tensor*) – The input, which should be (B, T) shaped.



* **Returns**

    
    * Final outputs of the encoding layer.


    * Encoded processing of the input tensor




* **Return type**

    Tuple[pt.Tensor, pt.Tensor]



#### loss_over_time(original: torch.Tensor, output: torch.Tensor)
Sum losses over time dimension, comparing original tokens and predicted tokens.


* **Parameters**

    
    * **original** (*pt.Tensor*) – The original input


    * **output** (*pt.Tensor*) – The predicted output



* **Returns**

    The aggregated CrossEntropyLoss.



* **Return type**

    float



#### mask_input(padded_input: torch.Tensor)
Creates a mask tensor to filter out padding.


* **Parameters**

    **padded_input** (*pt.Tensor*) – The padded input (B, T)



* **Returns**

    A boolean tensor (B, T).  True indicates the value at that time is usable, not padding.



* **Return type**

    pt.Tensor



#### predict_step(batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None)
> Calls _common_step for step ‘predict’.

batch: pt. Tuple[dict, pt.Tensor]

    An tuple of a metadata dictionary and the associated input data

batch_idx: int

    > The index of the batch.  Required to match the parent signature.  Unused in our model.

    dataloader_idx: int

        Index of the current dataloader.   Required to match the parent signature.  Unused in our model.

    Tuple[dict, pt.Tensor]

        An tuple of the batch metadata dictionary and the associated output data


#### test_step(x: torch.Tensor, batch_index: int)
Calls _common_step for step ‘val’.


* **Parameters**

    
    * **x** (*pt. Tensor*) – An input tensor


    * **batch_index** (*int*) – The index of the batch.  Required to match the parent signature.  Unused in our model.



* **Returns**

    Format expected by the parent class. The loss returned by _common_step.



* **Return type**

    float



#### training(_: boo_ )

#### training_step(x: torch.Tensor, batch_index: int)
Calls _common_step for step ‘train’.


* **Parameters**

    
    * **x** (*pt. Tensor*) – An input tensor


    * **batch_index** (*int*) – The index of the batch.  Required to match the parent signature.  Unused in our model.



* **Returns**

    Format expected by the parent class. Has a single key, ‘loss’ with the return value of _common_step.



* **Return type**

    dict



#### validation_step(x: torch.Tensor, batch_index: int)
Calls _common_step for step ‘val’.


* **Parameters**

    
    * **x** (*pt. Tensor*) – An input tensor


    * **batch_index** (*int*) – The index of the batch.  Required to match the parent signature.  Unused in our model.



* **Returns**

    Format expected by the parent class. The loss returned by _common_step.



* **Return type**

    float



### autoencoder.item_path(index: int, suffix: str = 'png', dir_only: bool = False, is_collection: bool = False)
A helper function to construct a file path string given a data item index.


* **Parameters**

    
    * **index** (*int*) – The index of the data item.


    * **suffix** (*str*) – The file suffix.


    * **dir_only** (*bool*) – Construct only a path to the enclosing directory.


    * **is_collection** – A data item index refers to a collection of files and not a single file.



* **Returns**

    The file path.



* **Return type**

    str
