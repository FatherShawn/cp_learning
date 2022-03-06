"""
A controller script for configuring and launching Pytorch Lightning's Trainer for the
QuackLatentClassifier model: cp_latent_classifier.QuackLatentClassifier().
"""
# You must import Comet before these modules: torch
# https://github.com/PyTorchLightning/pytorch-lightning/issues/5829.
import comet_ml
from time import gmtime, strftime
from pathlib import Path
from cp_latent_data import QuackLatentDataModule
from densenet import CensoredDataWriter
from cp_latent_classifier import QuackLatentClassifier
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from ray_lightning import RayPlugin
from argparse import ArgumentParser, Namespace


def main(args: Namespace) -> None:
    """
    The executable logic for this controller.

    For the training loop:
    - Instantiates a data object using `cp_latent_data.QuackLatentDataModule`.
    - Instantiates a model using `cp_latent_classifier.QuackLatentClassifier`.
    - Instantiates a strategy plugin using `ray_lightning.ray_ddp.RayPlugin`.
    - Instantiates callback objects:
    -- A logger using `pytorch_lightning.loggers.comet.CometLogger`
    -- A learning rate monitor using `pytorch_lightning.callbacks.lr_monitor.LearningRateMonitor`
    -- A checkpoint creator using `pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`
    -- An early stopping monitor using `pytorch_lightning.callbacks.early_stopping.EarlyStopping`
    Then using these objects, instantiates a training control object using `pytorch_lightning.trainer.trainer.Trainer`

    For inference with a trained model, just the logger and the ray strategy are used along with an instance of
    `densenet.CensoredDataWriter` which when composed with Trainer prepares the prediction loop to output its results
    to file on each iteration.

    Parameters
    ----------
    args: Namespace
         Command line arguments.  Possible arguments are:

         `--data_dir`
            *str* default='./data'  The top directory of the data storage tree.
         `--batch_size`
            *int* default=4 The batch size used for processing data.
         `--num_workers`
            *int* default=0 The number of worker processes used by the data loader.
         `--evaluate`
            *bool* Flag to output undetermined data from the inference loop. True when present, otherwise False
         `--checkpoint_path`
            *str* A checkpoint used for manual restart. Only the weights are used.
         `--storage_path`
            *str* default='./data/encoded' A path for storing the outputs from inference.
         `--l_rate`
            *float* default=1e-1 Hyperparameter passed to QuackAutoEncoder.
         `--l_rate_min`
            *float* default=1e-3 Hyperparameter passed to QuackAutoEncoder.
         `--l_rate_max_epoch`
            *int* default=-1 Hyperparameter passed to QuackAutoEncoder.
         `--exp_label`
            *str* default='autoencoder-train' Label passed to the logger.
         `--ray_nodes`
            *int* default=4 Number of parallel nodes passed to the Ray plugin.
         `--freeze`
            *bool* Flag to construct so that the image analyzing layers of the pre-trained Densenet are frozen for
            training.

    Returns
    -------
    void

    """
    data = QuackLatentDataModule(
        args.data_dir,
        batch_size=args.batch_size,
        workers=args.num_workers
    )
    model = QuackLatentClassifier(
            initial_size=256,
            learning_rate=args.l_rate,
            learning_rate_min=args.l_rate_min,
            lr_max_epochs=args.l_rate_max_epoch
    )
    if args.checkpoint_path is not None:
        model = QuackLatentClassifier.load_from_checkpoint(
            args.checkpoint_path,
            learning_rate=args.l_rate,
            learning_rate_min=args.l_rate_min,
            lr_max_epochs=args.l_rate_max_epoch
        )
    ray_plugin = RayPlugin(
        num_workers=args.ray_nodes,
        num_cpus_per_worker=1,
        use_gpu=False,
        find_unused_parameters=False
    )
    date_time = strftime("%d %b %Y %H:%M", gmtime())
    device_logger = DeviceStatsMonitor()
    checkpoint_storage = Path(args.storage_path)
    checkpoint_storage.mkdir(parents=True, exist_ok=True)
    # API configuration for comet: https://www.comet.ml/docs/python-sdk/advanced/#python-configuration
    comet_logger = CometLogger(
        project_name="censored-planet",
        experiment_name=f'{args.exp_label}: {date_time}',
    )
    if args.predict:
        writer_callback = CensoredDataWriter(
            write_interval='batch',
            storage_path=args.storage_path
        )
        trainer = Trainer.from_argparse_args(
            args,
            logger=comet_logger,
            callbacks=[writer_callback, device_logger],
            strategy=ray_plugin,
        )
        model.freeze()
        print('Ready for inference...')
        trainer.predict(model, datamodule=data, return_predictions=False)
        return
    else:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode='min',
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=True,
            filename='latent_checkpoint-{step}-{val_loss:02.2f}',
            dirpath=checkpoint_storage,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode='min',
            patience=10,
            stopping_threshold=0.05,
            check_finite=True,  # Stops training if the monitored metric becomes NaN or infinite.
        )
        trainer = Trainer.from_argparse_args(
            args,
            logger=comet_logger,
            callbacks=[early_stopping_callback, checkpoint_callback, device_logger, lr_monitor],
            weights_save_path=checkpoint_storage
        )
        print('Ready for training...')
        trainer.fit(model, datamodule=data)
        print('Post fit testing...')
        trainer.test(model, datamodule=data)


if __name__ == '__main__':
    # Add arguments to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str, default='./data')
    arg_parser.add_argument('--batch_size', type=int, default=4)
    arg_parser.add_argument('--num_workers', type=int, default=0)
    arg_parser.add_argument('--checkpoint_path', type=str)
    arg_parser.add_argument('--storage_path', type=str, default='./data/encoded')
    arg_parser.add_argument('--exp_label', type=str, default='latent-train')
    arg_parser.add_argument('--ray_nodes', type=int, default=4)
    arg_parser.add_argument('--l_rate', type=float, default=1e-1)
    arg_parser.add_argument('--l_rate_min', type=float, default=1e-3)
    arg_parser.add_argument('--l_rate_max_epoch', type=int, default=-1)
    arg_parser.add_argument('--freeze', action='store_true', default=False)
    arg_parser.add_argument('--predict', action='store_true', default=False)

    # add trainer arguments (gpus=x, precision=...)
    arg_parser = Trainer.add_argparse_args(arg_parser)
    arguments = arg_parser.parse_args()
    main(arguments)
