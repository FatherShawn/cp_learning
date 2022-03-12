# autoencoder module

The autoencoder class along with classes that compose the autoencoder class and helper functions.


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



### _class_ autoencoder.AutoencoderWriter(write_interval: str = 'batch', storage_path: str = '~/data', filtered: bool = False, evaluate: bool = False, reduction_threshold: float = 1.0)
Bases: `pytorch_lightning.callbacks.prediction_writer.BasePredictionWriter`

Extends pytorch_lightning.callbacks.prediction_writer.BasePredictionWriter to store encoded Quack data.


#### \__init__(write_interval: str = 'batch', storage_path: str = '~/data', filtered: bool = False, evaluate: bool = False, reduction_threshold: float = 1.0)
Constructor for AutoencoderWriter.


* **Parameters**

    
    * **write_interval** (*str*) – See parent class BasePredictionWriter


    * **storage_path** (*str*) – A string file path to the directory in which output will be stored.


    * **filtered** (*bool*) – Should the output be filtered to exclude undetermined items (censored/uncensored only)?


    * **evaluate** (*bool*) – Should the output be filtered to include only undetermined items for model evaluation?



#### write_on_batch_end(trainer: pytorch_lightning.trainer.trainer.Trainer, pl_module: pytorch_lightning.core.lightning.LightningModule, prediction: Any, batch_indices: Optional[Sequence[int]], batch: Any, batch_idx: int, dataloader_idx: int)
Logic to write the results of a single batch to files.


* **Parameters**

    **class.** (*Parameter signature defined in the parent*) – 



* **Return type**

    void



#### write_on_epoch_end(trainer: pytorch_lightning.trainer.trainer.Trainer, pl_module: pytorch_lightning.core.lightning.LightningModule, predictions: Sequence[Any], batch_indices: Optional[Sequence[Any]])
This class runs on every distributed node and aggregation is not practical due to the size of our dataset.  We
do not save the predictions after the batch to avoid running out of memory. Method is required but therefore
nothing to do here.


#### \__abstractmethods__(_ = frozenset({}_ )

#### \__module__(_ = 'autoencoder_ )

### _class_ autoencoder.AttentionScore(dim: int)
Bases: `torch.nn.modules.module.Module`

Uses the dot-product to calculate the attention scores.

### References

Edward Raff. 2021. Inside Deep Learning: Math, Algorithms, Models. Manning
Publications Co., Shelter Island, New York


#### \__init__(dim: int)
Constructs AttentionScore.


* **Parameters**

    **dim** (*int*) – The dimension of the hidden state axis coming into the dot-product.



#### forward(states: torch.Tensor, context: torch.Tensor)
Computes the dot-product score.


* **Parameters**

    
    * **states** (*pt.Tensor*) – Hidden states; shape (B, T, H)


    * **context** (*pt.Tensor*) – Context values; shape (B, H)



* **Returns**

    Scores for T items, based on context; shape (B, T, 1)



* **Return type**

    pt.Tensor



#### \__module__(_ = 'autoencoder_ )

#### training(_: boo_ )

### _class_ autoencoder.AttentionModule()
Bases: `torch.nn.modules.module.Module`

Applies attention to the hidden states.

### References

Edward Raff. 2021. Inside Deep Learning: Math, Algorithms, Models. Manning
Publications Co., Shelter Island, New York


#### \__init__()
Initializes internal Module state, shared by both nn.Module and ScriptModule.


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



#### \__module__(_ = 'autoencoder_ )

#### training(_: boo_ )

### _class_ autoencoder.QuackAutoEncoder(num_embeddings: int, embed_size: int, hidden_size: int, layers: int = 1, max_decode_length: Optional[int] = None, learning_rate: float = 0.1, learning_rate_min: float = 0.0001, lr_max_epochs: int = - 1, \*args: Any, \*\*kwargs: Any)
Bases: `pytorch_lightning.core.lightning.LightningModule`

A Sequence-to-Sequence based autoencoder

### References

Edward Raff. 2021. Inside Deep Learning: Math, Algorithms, Models. Manning
Publications Co., Shelter Island, New York


#### \__init__(num_embeddings: int, embed_size: int, hidden_size: int, layers: int = 1, max_decode_length: Optional[int] = None, learning_rate: float = 0.1, learning_rate_min: float = 0.0001, lr_max_epochs: int = - 1, \*args: Any, \*\*kwargs: Any)
Constructor for QuackAutoEncoder.


* **Parameters**

    
    * **num_embeddings** (*int*) – Hyperparameter for nn.Embedding


    * **embed_size** (*int*) – Hyperparameter for nn.Embedding and nn.GRU


    * **hidden_size** (*int*) – Hyperparameter for nn.GRU


    * **layers** (*int*) – Hyperparameter for nn.GRU


    * **max_decode_length** (*int*) – Hyperparameter used to limit the decoder module.


    * **learning_rate** (*float*) – Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR


    * **learning_rate_min** (*float*) – Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR


    * **lr_max_epochs** (*int*) – Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR


    * **args** (*Any*) – Passed to the parent constructor.


    * **Any** (*kwargs*) – Passed to the parent constructor.



#### mask_input(padded_input: torch.Tensor)
Creates a mask tensor to filter out padding.


* **Parameters**

    **padded_input** (*pt.Tensor*) – The padded input (B, T)



* **Returns**

    A boolean tensor (B, T).  True indicates the value at that time is usable, not padding.



* **Return type**

    pt.Tensor



#### loss_over_time(original: torch.Tensor, output: torch.Tensor)
Sum losses over time dimension, comparing original tokens and predicted tokens.


* **Parameters**

    
    * **original** (*pt.Tensor*) – The original input


    * **output** (*pt.Tensor*) – The predicted output



* **Returns**

    The aggregated CrossEntropyLoss.



* **Return type**

    float



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



#### test_step(x: torch.Tensor, batch_index: int)
Calls _common_step for step ‘val’.


* **Parameters**

    
    * **x** (*pt. Tensor*) – An input tensor


    * **batch_index** (*int*) – The index of the batch.  Required to match the parent signature.  Unused in our model.



* **Returns**

    Format expected by the parent class. The loss returned by _common_step.



* **Return type**

    float



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



#### configure_optimizers()
Configures the optimizer and learning rate scheduler objects.


* **Returns**

    A dictionary with keys:


    * optimizer: pt.optim.AdamW


    * lr_scheduler: pt.optim.lr_scheduler.CosineAnnealingLR




* **Return type**

    dict



#### \__module__(_ = 'autoencoder_ )

#### training(_: boo_ )
