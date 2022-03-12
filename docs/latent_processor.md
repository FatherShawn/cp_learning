# latent_processor module

A controller script for configuring and launching Pytorch Lightning’s Trainer for the
QuackLatentClassifier model: cp_latent_classifier.QuackLatentClassifier().


### latent_processor.main(args: argparse.Namespace)
The executable logic for this controller.

For the training loop:
- Instantiates a data object using `cp_latent_data.QuackLatentDataModule`.
- Instantiates a model using `cp_latent_classifier.QuackLatentClassifier`.
- Instantiates a strategy plugin using `ray_lightning.ray_ddp.RayPlugin`.
- Instantiates callback objects:

> 
> * A logger using `pytorch_lightning.loggers.comet.CometLogger`


> * A learning rate monitor using `pytorch_lightning.callbacks.lr_monitor.LearningRateMonitor`


> * A checkpoint creator using `pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`


> * An early stopping monitor using `pytorch_lightning.callbacks.early_stopping.EarlyStopping`

Then using these objects, instantiates a training control object using `pytorch_lightning.trainer.trainer.Trainer`

For inference with a trained model, just the logger and the ray strategy are used along with an instance of
`densenet.CensoredDataWriter` which when composed with Trainer prepares the prediction loop to output its results
to file on each iteration.


* **Parameters**

    **args** (*Namespace*) – Command line arguments.  Possible arguments are:




* **Return type**

    void
