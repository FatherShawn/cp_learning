# ae_processor module

A controller script for configuring and launching Pytorch Lightning’s Trainer for the Sequence-to-Sequence
Autoencoder: autoencoder.QuackAutoEncoder().


### ae_processor.main(args: argparse.Namespace)
The executable logic for this controller.

For the training loop:


* Instantiates a data object using cp_tokenized_data.QuackTokenizedDataModule.


* Instantiates a model using autoencoder.QuackAutoEncoder.


* Instantiates a strategy plugin using ray_lightning.ray_ddp.RayPlugin.


* Instantiates callback objects:

– A logger using pytorch_lightning.loggers.comet.CometLogger
– A learning rate monitor using pytorch_lightning.callbacks.lr_monitor.LearningRateMonitor
– A checkpoint creator using pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
– An early stopping monitor using pytorch_lightning.callbacks.early_stopping.EarlyStopping

Then using these objects, instantiates a training control object using pytorch_lightning.trainer.trainer.Trainer

For inference with a trained model, just the logger and the ray strategy are used along with an instance of
autoencoder.AutoencoderWriter which when composed with Trainer prepares the prediction loop to output its results
to file on each iteration.


* **Parameters**

    **args** (*Namespace*) – Command line arguments.  Possible arguments are:

    –data_dir

        *str* default=’./data’  The top directory of the data storage tree.

    –batch_size

        *int* default=4 The batch size used for processing data.

    –num_workers

        *int* default=0 The number of worker processes used by the data loader.

    –embed_size

        *int* default=128 Hyperparameter passed to QuackAutoEncoder.

    –hidden_size

        *int* default=512 Hyperparameter passed to QuackAutoEncoder.

    –encode

        *bool* Flag to run the inference loop instead of train. True when present, otherwise False

    –filtered

        *bool* Flag to output labeled data from the inference loop. True when present, otherwise False

    –evaluate

        *bool* Flag to output undetermined data from the inference loop. True when present, otherwise False

    –checkpoint_path

        *str* A checkpoint used for manual restart. Only the weights are used.

    –storage_path

        *str* default=’./data/encoded’ A path for storing the outputs from inference.

    –l_rate

        *float* default=1e-1 Hyperparameter passed to QuackAutoEncoder.

    –l_rate_min

        *float* default=1e-3 Hyperparameter passed to QuackAutoEncoder.

    –l_rate_max_epoch

        *int* default=-1 Hyperparameter passed to QuackAutoEncoder.

    –exp_label

        *str* default=’autoencoder-train’ Label passed to the logger.

    –ray_nodes

        *int* default=4 Number of parallel nodes passed to the Ray plugin.




* **Return type**

    void
