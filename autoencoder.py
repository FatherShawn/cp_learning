from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class QuackAutoEncoder(pl.LightningModule):

    def __init__(self, data_width: int, encoded_width: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = nn.GRU(input_size=data_width, hidden_size=encoded_width, num_layers=2, batch_first=True)
        self.decoder = nn.GRU(input_size=encoded_width, hidden_size=data_width, num_layers=2, batch_first=True)
        self.learning_rate = 1e-3

    def forward(self, x) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        return encoded

    def _common_step(self, x: torch.Tensor, batch_index: int, step_id: str) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log(f"{step_id}_loss", loss)
        return loss

    def training_step(self, x: torch.Tensor, batch_index: int) -> torch.Tensor:
        return self._common_step(x, batch_index, 'train')

    def validation_step(self, x: torch.Tensor, batch_index: int) -> torch.Tensor:
        return self._common_step(x, batch_index, 'val')

    def test_step(self, x: torch.Tensor, batch_index: int) -> torch.Tensor:
        return self._common_step(x, batch_index, 'test')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)