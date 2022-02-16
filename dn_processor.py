# You must import Comet before these modules: torch
# https://github.com/PyTorchLightning/pytorch-lightning/issues/5829.
import comet_ml
from time import gmtime, strftime
from pathlib import Path
from cp_image_data import QuackImageDataModule
from densenet import QuackDenseNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from ray_lightning import RayPlugin
from argparse import ArgumentParser, Namespace


def main(args: Namespace) -> None:
    data = QuackImageDataModule(
        args.data_dir,
        batch_size=args.batch_size,
        workers=args.num_workers,
        simple_transforms=args.simple_transforms
    )
    model = QuackDenseNet(
            learning_rate=args.l_rate,
            learning_rate_min=args.l_rate_min,
            lr_max_epochs=args.l_rate_max_epoch,
            freeze=args.freeze
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
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # API configuration for comet: https://www.comet.ml/docs/python-sdk/advanced/#python-configuration
    # We have to instantiate by case if we want experiment names by case, due to CometLogger architecture.
    comet_logger = CometLogger(
        project_name="censored-planet",
        experiment_name=f'{args.exp_label}: {date_time}',
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_train_steps=2000,
        auto_insert_metric_name=True,
        filename='checkpoint_{epoch:02d}-{step}-{val_loss:02.2f}',
        dirpath=checkpoint_storage,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode='min',
        patience=10,
        stopping_threshold=0.2,
        check_finite=True,  # Stops training if the monitored metric becomes NaN or infinite.
    )
    trainer = Trainer.from_argparse_args(
        args,
        logger=comet_logger,
        callbacks=[early_stopping_callback, checkpoint_callback, device_logger, lr_monitor],
        plugins=[ray_plugin],
        weights_save_path=checkpoint_storage
    )
    print('Ready for training...')
    if args.checkpoint_path is not None:
        trainer.fit(model, datamodule=data, ckpt_path=args.checkpoint_path)
    else:
        trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    # Add arguments to make a more flexible cli tool.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str, default='./data')
    arg_parser.add_argument('--batch_size', type=int, default=4)
    arg_parser.add_argument('--num_workers', type=int, default=0)
    arg_parser.add_argument('--checkpoint_path', type=str)
    arg_parser.add_argument('--comet_storage', type=str, default='.')
    arg_parser.add_argument('--storage_path', type=str, default='./data/encoded')
    arg_parser.add_argument('--exp_label', type=str, default='autoencoder-train')
    arg_parser.add_argument('--ray_nodes', type=int, default=4)
    arg_parser.add_argument('--l_rate', type=float, default=1e-1)
    arg_parser.add_argument('--l_rate_min', type=float, default=1e-3)
    arg_parser.add_argument('--l_rate_max_epoch', type=int, default=-1)
    arg_parser.add_argument('--freeze', action='store_true', default=False)
    arg_parser.add_argument('--simple_transforms', action='store_true', default=False)

    # add trainer arguments (gpus=x, precision=...)
    arg_parser = Trainer.add_argparse_args(arg_parser)
    arguments = arg_parser.parse_args()
    main(arguments)