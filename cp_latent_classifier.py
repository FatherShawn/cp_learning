from typing import Any, Tuple
import torch as pt
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from cp_flatten import QuackConstants
import torch.nn.functional as F
import pytorch_lightning as pl

class QuackLatentClassifier(pl.LightningModule):

    def __init__(self, initial_size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        leak_rate = 0.1  # Something in the range [0.01, 0.3].
        self.__model = nn.Sequential(
            nn.Linear(initial_size, initial_size // 4),
            nn.LeakyReLU(leak_rate),
            nn.Linear(initial_size // 4, initial_size // 4),
            nn.LeakyReLU(leak_rate),
            nn.Linear(initial_size // 4, initial_size // 8),
            nn.LeakyReLU(leak_rate),
            nn.Linear(initial_size // 8, initial_size // 1),
        )

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        return self.__model(x)

    def _common_step(self, data: Tuple[pt.Tensor, int], batch_index: int, step_id: str) -> float:
        value, label = data
        prediction = self.forward(value)


    def training_step(self, x: pt.Tensor, batch_index: int) -> float:
        return self._common_step(x, batch_index, 'train')

    def validation_step(self, x: pt.Tensor, batch_index: int) -> float:
        return self._common_step(x, batch_index, 'val')

    def test_step(self, x: pt.Tensor, batch_index: int) -> float:
        return self._common_step(x, batch_index, 'test')

    def configure_optimizers(self) -> pt.optim.Optimizer:
        return pt.optim.AdamW(self.parameters(), lr=self.__learning_rate)