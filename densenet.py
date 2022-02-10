from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from typing import Any, List, Optional, TypedDict, Tuple, Dict


import torch as pt
from torch import nn
from torchvision import models
import pytorch_lightning as pl
import torchmetrics as tm


class QuackMetric(TypedDict):
    accuracy: tm.Accuracy
    f1: tm.F1Score
    auroc: tm.AUROC


class QuackMetricSet(TypedDict):
    train: QuackMetric
    val: QuackMetric

class QuackDenseNet(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
        train_metric = QuackMetric(
            accuracy=tm.Accuracy(),
            f1=tm.F1Score(num_classes=2),
            auroc=tm.AUROC(num_classes=2)
        )
        val_metric = QuackMetric(
            accuracy=tm.Accuracy(),
            f1=tm.F1Score(num_classes=2),
            auroc=tm.AUROC(num_classes=2)
        )
        self.__metrics = QuackMetricSet(
            train=train_metric,
            val=val_metric
        )
        self.__bce = nn.BCELoss()

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
        return self.__densenet(x)

    def _common_step(self, x: Tuple[pt.Tensor, pt.Tensor], batch_index: int, step_id: str) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        inputs, labels = x
        outputs = self.forward(inputs)
        output_labels = outputs.ge(0.5).long()  # Binarize predictions to 0 and 1
        loss = self.__bce(outputs, labels)
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
        inputs, _ = batch
        return self.forward(inputs)

    def training_step_end(self, outputs: dict, *args, **kwargs):
        self.__metrics['train']['accuracy'](outputs['predicted'], outputs['expected'])
        self.log('train_acc', self.__metrics['train']['accuracy'])
        self.__metrics['train']['f1'](outputs['predicted'], outputs['expected'])
        self.log('train_f1', self.__metrics['train']['f1'])
        self.__metrics['train']['auroc'](outputs['predicted'], outputs['expected'])
        self.log('train_auroc',  self.__metrics['train']['auroc'])

    def validation_step_end(self, outputs: dict, *args, **kwargs):
        self.__metrics['val']['accuracy'].update(outputs['predicted'], outputs['expected'])
        self.__metrics['val']['f1'].update(outputs['predicted'], outputs['expected'])
        self.__metrics['train']['auroc'].update(outputs['predicted'], outputs['expected'])

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        super().validation_epoch_end(outputs)



