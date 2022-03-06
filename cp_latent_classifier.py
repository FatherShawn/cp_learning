"""
The latent classifier model with classes composed into the classifier class.
"""
from typing import Any, Tuple, List, Optional
import torch as pt
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torchmetrics as tm


class QuackLatentClassifier(pl.LightningModule):
    """
    A binary classifier which operates on encoded tensors produced by
    autoencoder.QuackAutoEncoder.
    """
    def __init__(self, initial_size: int, learning_rate: float = 1e-1, learning_rate_min: float = 1e-4,
                 lr_max_epochs: int = -1, *args: Any, **kwargs: Any) -> None:
        """
       Constructor for QuackLatentClassifier.

       Parameters
       ----------
       learning_rate: float
           Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR
       learning_rate_min: float
           Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR
       lr_max_epochs: int
           Hyperparameter passed to pt.optim.lr_scheduler.CosineAnnealingLR
       args: Any
           Passed to the parent constructor.
       kwargs: Any
           Passed to the parent constructor.
       """
        super().__init__(*args, **kwargs)
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
        leak_rate = 0.1  # Something in the range [0.01, 0.3].
        self.__model = nn.Sequential(
            nn.Linear(initial_size, initial_size // 4),
            nn.LeakyReLU(leak_rate),
            nn.Linear(initial_size // 4, initial_size // 8),
            nn.LeakyReLU(leak_rate),
            nn.Linear(initial_size // 8, 1)
        )
        self.__loss_module = nn.BCEWithLogitsLoss()

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        """
        Process a batch of input through the model.

        Parameters
        ----------
        x: pt.Tensor
            The input

        Returns
        -------
        pt.Tensor
            The output, which should be (B, 1) sized, of single probability floats.

        """
        return self.__model(x)

    def _common_step(self, data: Tuple[pt.Tensor, pt.Tensor], batch_index: int, step_id: str) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """
        The step task in each loop type shares a common set of tasks.

        Parameters
        ----------
        data: Tuple[pt.Tensor, pt.Tensor]
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
        values, labels = data
        prediction = self.forward(values)  # Shape (B, 1)
        loss = self.__loss_module(prediction, labels)
        # Binarize predictions to 0 and 1.
        prediction = self.__to_probability(prediction)
        prediction_labels = prediction.ge(0.5).long()
        log_interval_option = None if step_id == 'train' else True
        log_sync = False if step_id == 'train' else True
        self.log(f"{step_id}_loss", loss, on_step=log_interval_option, sync_dist=log_sync)
        # Return labels and output_labels for use in accuracy, which expects integer tensors.
        return loss, labels.to(pt.int8), prediction_labels.to(pt.int8)

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

    def predict_step(self, batch: Tuple[pt.Tensor, List[dict]], batch_idx: int, dataloader_idx: Optional[int] = None) -> \
    Tuple[List[dict], pt.Tensor]:
        """
        Calls `forward` for prediction.

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
        # Classifier outputs an un-normalized confidence score.
        # Use sigmoid to transform to a probability.
        confidence = self.forward(inputs)
        output = self.__to_probability(confidence)
        return meta, output

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
        configured_optimizer = pt.optim.AdamW(params=parameters, lr=self.__learning_rate_init)
        return {
            'optimizer': configured_optimizer,
            'lr_scheduler': pt.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=configured_optimizer,
                T_max=self.__lr_max_epochs,
                eta_min=self.__learning_rate_min
            )
        }
