"""
The densenet model with classes composed into the densenet class.
"""
import json
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pytorch_lightning.utilities.distributed import rank_zero_info
from typing import Any, List, Optional, Sequence, Tuple
from pathlib import Path
import torch as pt
from torch import nn
from torchvision import models
import pytorch_lightning as pl
import torchmetrics as tm
from pytorch_lightning.callbacks import BasePredictionWriter


class CensoredDataWriter(BasePredictionWriter):
    """
    Extends pytorch_lightning.callbacks.prediction_writer.BasePredictionWriter
    to store metadata for detected censorship.
    """

    def __init__(self, write_interval: str = "batch", storage_path: str = '~/data') -> None:
        """
               Constructor for AutoencoderWriter.

               Parameters
               ----------
               write_interval: str
                   See parent class BasePredictionWriter
               storage_path: str
                   A string file path to the directory in which output will be stored.
        """
        super().__init__(write_interval)
        self.__storage_path = storage_path

    def write_on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", prediction: Any,
                           batch_indices: Optional[Sequence[int]], batch: Any, batch_idx: int,
                           dataloader_idx: int) -> None:
        """
        Logic to write the results of a single batch to files.

        Parameters
        ----------
        Parameter signature defined in the parent class.

        Returns
        ----------
        void

        See Also
        ----------
        pytorch_lightning.callbacks.prediction_writer.BasePredictionWriter.write_on_batch_end
        """
        meta: List[dict]
        processed: pt.Tensor
        meta, processed = batch
        # Copy to cpu and convert to numpy array.
        prep_for_numpy = processed.cpu()
        data = prep_for_numpy.numpy()
        for outcome in data:
            # Ensure storage is ready.
            storage_path = Path(self.__storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            data_storage = Path(self.__storage_path + 'densenet_detections.txt')
            # Get metadata for this outcome.
            outcome_meta = meta.pop(0)
            if outcome >= 0.5:
                # Predicted as censored.
                with data_storage.open(mode='a') as target:
                    json.dump(outcome_meta, target)


class QuackDenseNet(pl.LightningModule):
    """
    A modification of the Pytorch Densenet 121 pretrained model.

    References
    ----------
    https://pytorch.org/hub/pytorch_vision_densenet/
    """
    def __init__(self, learning_rate: float = 1e-1, learning_rate_min: float = 1e-4,
                 lr_max_epochs: int = -1, freeze: bool = True, *args: Any, **kwargs: Any) -> None:
        """
        Constructor for QuackDenseNet.

        Parameters
        ----------
        learning_rate: float
            Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR
        learning_rate_min: float
            Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR
        lr_max_epochs: int
            Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR
        freeze: bool
            Should the image analyzing layers of the pre-trained Densenet be frozen?
        args: Any
            Passed to the parent constructor.
        kwargs: Any
            Passed to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        # Load the pre-trained densenet
        pre_trained = models.densenet121(pretrained=True)
        if freeze:
            # Freeze the existing gradients.
            pre_trained.requires_grad_(False)
        # We want to replace the classifier.  New instances of models have
        # requires_grad = True by default.
        classifier_features_in = pre_trained.classifier.in_features
        pre_trained.classifier = nn.Linear(classifier_features_in, 1)
        self.__densenet = pre_trained
        self.__to_probability = nn.Sigmoid()
        self.__learning_rate_init = learning_rate
        self.__learning_rate_min = learning_rate_min
        self.__lr_max_epochs = lr_max_epochs
        self.__train_acc = tm.Accuracy()
        self.__train_f1 = tm.F1Score(num_classes=2)
        self.__val_acc = tm.Accuracy()
        self.__val_f1 = tm.F1Score(num_classes=2)
        self.__test_acc = tm.Accuracy()
        self.__test_f1 = tm.F1Score(num_classes=2)
        # We have 653481 uncensored (negative samples) and 215016 positive samples in our dataset.
        # 653481 / 215016 = 3.0392203371
        balance_factor = pt.tensor(3.039)
        self.__loss_module = nn.BCEWithLogitsLoss(pos_weight=balance_factor)
        # For tuning.
        self.batch_size = 2

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        """
        Process a batch of input through the model.

        Parameters
        ----------
        x: pt.Tensor
            The input, which should be (B, 3, H, W) shaped, where:
            B: batch size
            3: Densnet is trained on RGB = 3 channels
            H: height
            W: width

        Returns
        -------
        pt.Tensor
            The output, which should be (B, 1) sized, of single probability floats.

        """
        # Densenet outputs an un-normalized confidence score.
        # Use sigmoid to transform to a probability.
        confidence = self.__densenet(x)
        return self.__to_probability(confidence)

    def _common_step(self, x: Tuple[pt.Tensor, pt.Tensor], batch_index: int, step_id: str) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """
        The step task in each loop type shares a common set of tasks.

        Parameters
        ----------
        x: Tuple[pt.Tensor, pt.Tensor]
            The batch
        batch_index: int
            The batch index
        step_id: str
            The step id.

        Returns
        -------
        Tuple[pt.Tensor, pt.Tensor, pt.Tensor]
            A tuple of batch losses, batch labels as integers, batch predictions as integers
            each in a tensor.
        """
        inputs, labels = x
        outputs = self.forward(inputs)
        # Binarize predictions to 0 and 1.
        output_labels = outputs.ge(0.5).long()
        # Then match to labels type.
        output_labels = output_labels.to(pt.float)
        loss = self.__loss_module(outputs, labels)
        log_interval_option = None if step_id == 'train' else True
        log_sync = False if step_id == 'train' else True
        self.log(f"{step_id}_loss", loss, on_step=log_interval_option, sync_dist=log_sync)
        # Return labels and output_labels for use in accuracy, which expects integer tensors.
        return loss, labels.to(pt.int8), output_labels.to(pt.int8)

    def training_step(self, x: Tuple[pt.Tensor, pt.Tensor], batch_index: int) -> dict:
        """
        Calls _common_step for step 'train'.

        Parameters
        ----------
        x: Tuple[pt.Tensor, pt.Tensor]
            The input tensor and a label tensor
        batch_index: int
            The index of the batch.  Required to match the parent signature.  Unused in our model.

        Returns
        -------
        dict
            Format expected by the parent class. Has three keys:

            loss
                The loss returned by `_common_step`.
            expected
                The labels from the batch returned by `_common_step`.
            predicted
                The predicted labels from the batch returned by `_common_step`.
        """
        loss, expected, predicted = self._common_step(x, batch_index, 'train')
        return {'loss': loss, 'expected': expected, 'predicted': predicted}

    def validation_step(self, x: Tuple[pt.Tensor, pt.Tensor], batch_index: int) -> dict:
        """
        Calls _common_step for step 'val'.

        Parameters
        ----------
        x: Tuple[pt.Tensor, pt.Tensor]
            The input tensor and a label tensor
        batch_index: int
            The index of the batch.  Required to match the parent signature.  Unused in our model.

        Returns
        -------
        dict
            Format expected by the parent class. Has three keys:

            loss
                The loss returned by `_common_step`.
            expected
                The labels from the batch returned by `_common_step`.
            predicted
                The predicted labels from the batch returned by `_common_step`.
        """
        loss, expected, predicted = self._common_step(x, batch_index, 'val')
        return {'loss': loss, 'expected': expected, 'predicted': predicted}

    def test_step(self, x: Tuple[pt.Tensor, pt.Tensor], batch_index: int) -> dict:
        """
        Calls _common_step for step 'test'.

        Parameters
        ----------
        x: Tuple[pt.Tensor, pt.Tensor]
            The input tensor and a label tensor
        batch_index: int
            The index of the batch.  Required to match the parent signature.  Unused in our model.

        Returns
        -------
        dict
            Format expected by the parent class. Has three keys:

            loss
                The loss returned by `_common_step`.
            expected
                The labels from the batch returned by `_common_step`.
            predicted
                The predicted labels from the batch returned by `_common_step`.
        """
        loss, expected, predicted = self._common_step(x, batch_index, 'test')
        return {'loss': loss, 'expected': expected, 'predicted': predicted}

    def predict_step(self, batch: Tuple[pt.Tensor, List[dict]], batch_idx: int, dataloader_idx: Optional[int] = None) -> Tuple[List[dict], pt.Tensor]:
        """
        Calls _common_step for step 'predict'.

        Parameters
        ----------
        batch: Tuple[pt.Tensor, List[dict]]
            An tuple of a metadata dictionary and the associated input data
        batch_idx: int
            The index of the batch.  Required to match the parent signature.  Unused in our model.
        dataloader_idx: int
            Index of the current dataloader.   Required to match the parent signature.  Unused in our model.

        Returns
        -------
        Tuple[dict, pt.Tensor]
            An tuple of the batch metadata dictionary and the associated output data
        """
        inputs, meta = batch
        return meta, self.forward(inputs)

    def training_step_end(self, outputs: dict, *args, **kwargs):
        """
        When using distributed backends, only a portion of the batch is inside the `training_step`.
        We calculate metrics here with the entire batch.

        Parameters
        ----------
        outputs: dict
            The return values from `training_step` for each batch part.
        args: Any
            Matching to the parent constructor.
        kwargs: Any
            Matching to the parent constructor.

        Returns
        -------
        void
        """
        self.__train_acc(outputs['predicted'], outputs['expected'])
        self.log('train_acc', self.__train_acc)
        self.__train_f1(outputs['predicted'], outputs['expected'])
        self.log('train_f1', self.__train_f1)

    def validation_step_end(self, outputs: dict, *args, **kwargs):
        """
        When using distributed backends, only a portion of the batch is inside the `validation_step`.
         We calculate metrics here with the entire batch.

        Parameters
        ----------
        outputs: dict
            The return values from `training_step` for each batch part.
        args: Any
            Matching to the parent constructor.
        kwargs: Any
            Matching to the parent constructor.

        Returns
        -------
        void
        """
        self.__val_acc.update(outputs['predicted'], outputs['expected'])
        self.__val_f1.update(outputs['predicted'], outputs['expected'])

    def test_step_end(self, outputs: dict, *args, **kwargs):
        """
        When using distributed backends, only a portion of the batch is inside the `test_step`.
         We calculate metrics here with the entire batch.

        Parameters
        ----------
        outputs: dict
            The return values from `training_step` for each batch part.
        args: Any
            Matching to the parent constructor.
        kwargs: Any
            Matching to the parent constructor.

        Returns
        -------
        void
        """
        self.__test_acc.update(outputs['predicted'], outputs['expected'])
        self.__test_f1.update(outputs['predicted'], outputs['expected'])

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """
        Called at the end of a test epoch with the output of all test steps.

        Now that all the test steps are complete, we compute the metrics.

        Parameters
        ----------
        outputs: None
            No outputs are passed on from `test_step_end`.

        Returns
        -------
        void
        """
        self.__test_acc.compute()
        self.__test_f1.compute()
        self.log('test_acc', self.__test_acc)
        self.log('test_f1', self.__test_f1)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """
        Called at the end of a validation epoch with the output of all test steps.

        Now that all the validation steps are complete, we compute the metrics.

        Parameters
        ----------
        outputs: None
            No outputs are passed on from `test_step_end`.

        Returns
        -------
        void
        """
        self.__val_acc.compute()
        self.__val_f1.compute()
        self.log('val_acc', self.__val_acc)
        self.log('val_f1', self.__val_f1)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler objects.

        Returns
        -------
        dict
            A dictionary with keys:

            - optimizer: pt.optim.AdamW
            - lr_scheduler: pt.optim.lr_scheduler.CosineAnnealingLR

        See Also
        --------
        pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        """
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        configured_optimizer = pt.optim.AdamW(params=trainable_parameters, lr=self.__learning_rate_init)
        return {
            'optimizer': configured_optimizer,
            'lr_scheduler': pt.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=configured_optimizer,
                T_max=self.__lr_max_epochs,
                eta_min=self.__learning_rate_min
            )
        }




