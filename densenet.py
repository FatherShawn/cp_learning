from typing import Any, List, Optional, Sequence, Tuple, Dict


import torch as pt
from torch import nn
from torchvision import models
import numpy as np
from cp_flatten import QuackConstants
import torch.nn.functional as F
import pytorch_lightning as pl


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

    def _common_step(self, x: Tuple[pt.Tensor, pt.Tensor], batch_index: int, step_id: str) -> pt.Tensor:
        inputs, labels = x
        cross_entropy = nn.CrossEntropyLoss()
        outputs = self.forward(inputs)
        loss = cross_entropy(outputs, labels)
        log_interval_option = None if step_id == 'train' else True
        log_sync = False if step_id == 'train' else True
        self.log(f"{step_id}_loss", loss, on_step=log_interval_option, sync_dist=log_sync)
        return loss

    def training_step(self, x: pt.Tensor, batch_index: int) -> dict:
        return {'loss': self._common_step(x, batch_index, 'train')}

    def validation_step(self, x: pt.Tensor, batch_index: int) -> float:
        return self._common_step(x, batch_index, 'val')

    def test_step(self, x: pt.Tensor, batch_index: int) -> float:
        return self._common_step(x, batch_index, 'test')

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tuple[List[dict], pt.Tensor]:
        inputs, _ = batch
        return self.forward(inputs)

    def configure_optimizers(self) -> Dict:
        configured_optimizer = pt.optim.AdamW(self.parameters(), lr=self.__learning_rate_init)
        return {
            'optimizer': configured_optimizer,
            'lr_scheduler': pt.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=configured_optimizer,
                T_max=self.__lr_max_epochs,
                eta_min=self.__learning_rate_min
            )
        }