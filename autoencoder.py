import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

def gru_block(input_dim, encoded_dim) -> nn.Sequential:
    pass


class QuackAutoencoder(pl.LightningModule):
    def __init__(self):
        pass

    def forward(self, *args, **kwargs) -> Any:
        pass

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        pass

    def configure_optimizers(self):
        pass