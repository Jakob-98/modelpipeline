import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torchmetrics
import numpy as np
import io
import torchvision

class Experiment(pl.LightningModule):
    def __init__(self, model, loss) -> None:
        super().__init__()

        self.model = model
        self.loss = loss
        # self.val_confusion = ConfusionMatrix(num_classes=6) #self._config.n_clusters)
        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1(number_classes=10,
        average="micro")
        self.train_auroc = torchmetrics.AUROC(number_classes=10,
        average="micro")
        self.val_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1(number_classes=10,
        average="micro")
        self.val_auroc = torchmetrics.AUROC(number_classes=10,
        average="micro")
    
    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        loss, pred_labels, true_labels, y, y_hat = self._shared_eval_step(batch, batch_idx)
        self._update_metrics(train=True, ytrue=true_labels, ypred=pred_labels)
        return loss
    
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        loss, pred_labels, true_labels, y, y_hat = self._shared_eval_step(batch, batch_idx)
        self._update_metrics(train=False, ytrue=true_labels, ypred=pred_labels)        
        return loss

    def _shared_eval_step(self, batch, batchidx):
        x1, x2, z, y = batch
        y_hat = self.model(x1,x2)
        loss = self.loss(y_hat, y)
        pred_labels = torch.argmax(y_hat, axis=1)
        true_labels = torch.argmax(y, axis=1)
        return loss, pred_labels, true_labels, y_hat, y

    def _update_metrics(self, train: bool, ytrue, ypred):
        if train:
            self.train_acc.update(ypred, ytrue)
            self.train_f1.update(ypred, ytrue)
            self.train_auroc.update(ypred, ytrue)
        else:
            self.val_acc.update(ypred, ytrue)
            self.val_f1.update(ypred, ytrue)
            self.val_auroc.update(ypred, ytrue)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = self.val_acc.compute()
        avg_f1 = self.val_f1.compute()
        avg_auroc = self.val_auroc.compute()
        # reset metrics
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        # log metrics\
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", avg_acc, prog_bar=True)
        self.log("val_f1", avg_f1, prog_bar=True)
        self.log("val_auroc", avg_auroc, prog_bar=True)
        return {"val_loss": avg_loss, "val_acc": avg_acc, "val_f1": avg_f1, "val_auroc": avg_auroc}


    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = self.train_acc.compute()
        avg_f1 = self.train_f1.compute()
        avg_auroc = self.train_auroc.compute()
        # reset metrics
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_auroc.reset()
        # log metrics
        self.log("train_loss", avg_loss, prog_bar=True)
        self.log("train_acc", avg_acc, prog_bar=True)
        self.log("train_f1", avg_f1, prog_bar=True)
        self.log("train_auroc", avg_auroc, prog_bar=True)
        return {"train_loss": avg_loss, "train_acc": avg_acc, "train_f1": avg_f1, "train_auroc": avg_auroc}

    def test_step(self, batch, batch_idx, optimizer_idx = 0):
        x, z, y = batch
        y_hat = self.model(x,z )
        loss = self.loss(y_hat, y)
        return {"loss": loss, "pred": y_hat} 
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)