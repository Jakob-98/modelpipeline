import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

class Experiment(pl.LightningModule):
    def __init__(self, model, loss) -> None:
        super().__init__()

        self.model = model
        self.loss = loss
    
    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)