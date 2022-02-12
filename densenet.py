from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pytorch_lightning.utilities.distributed import rank_zero_info
from typing import Any, List, Optional, TypedDict, Tuple
import torch as pt
from torch import nn
from torchvision import models
import pytorch_lightning as pl
import torchmetrics as tm


class QuackMetric(TypedDict):
    accuracy: tm.Accuracy
    f1: tm.F1Score


class QuackDenseNet(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-1, learning_rate_min: float = 1e-4,
                 lr_max_epochs: int = -1, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Load the pre-trained densenet
        pre_trained = models.densenet121(pretrained=True)
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
        self.__train_metric = QuackMetric(
            accuracy=tm.Accuracy(),
            f1=tm.F1Score(num_classes=2)
        )
        self.__val_metric = QuackMetric(
            accuracy=tm.Accuracy(),
            f1=tm.F1Score(num_classes=2)
        )
        # We have 653481 uncensored (negative samples) and 215016 positive samples in our dataset.
        # 653481 / 215016 = 3.0392203371
        balance_factor = pt.tensor(3.039)
        self.__loss_module = nn.BCEWithLogitsLoss(pos_weight=balance_factor)
        # For tuning.
        self.batch_size = 2

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        """

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
        return loss, labels, output_labels

    def training_step(self, x: pt.Tensor, batch_index: int) -> dict:
        loss, expected, predicted = self._common_step(x, batch_index, 'train')
        return {'loss': loss, 'expected': expected, 'predicted': predicted}

    def validation_step(self, x: pt.Tensor, batch_index: int) -> dict:
        loss, expected, predicted = self._common_step(x, batch_index, 'val')
        return {'loss': loss, 'expected': expected, 'predicted': predicted}

    def test_step(self, x: pt.Tensor, batch_index: int) -> dict:
        loss, expected, predicted = self._common_step(x, batch_index, 'test')
        return {'loss': loss, 'expected': expected, 'predicted': predicted}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tuple[List[dict], pt.Tensor]:
        inputs, meta = batch
        return self.forward(inputs), meta

    def training_step_end(self, outputs: dict, *args, **kwargs):
        self.__train_metric['accuracy'](outputs['predicted'], outputs['expected'])
        self.log('train_acc', self.__train_metric['accuracy'])
        self.__train_metric['f1'](outputs['predicted'], outputs['expected'])
        self.log('train_f1', self.__train_metric['f1'])

    def validation_step_end(self, outputs: dict, *args, **kwargs):
        self.__val_metric['accuracy'].update(outputs['predicted'], outputs['expected'])
        self.__val_metric['f1'].update(outputs['predicted'], outputs['expected'])

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.__val_metric['accuracy'].compute()
        self.__val_metric['f1'].compute()
        self.log('val_acc', self.__val_metric['accuracy'])
        self.log('val_f1', self.__val_metric['f1'])

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        configured_optimizer = pt.optim.AdamW(trainable_parameters, lr=self.__learning_rate_init)
        return {
            'optimizer': configured_optimizer,
            'lr_scheduler': pt.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=configured_optimizer,
                T_max=self.__lr_max_epochs,
                eta_min=self.__learning_rate_min
            )
        }




