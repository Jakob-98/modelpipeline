import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
import numpy as np
import io
import torchvision
import torchmetrics
import wandb
import seaborn as sns

class Experiment(pl.LightningModule):
    def __init__(self, model, loss, n_classes) -> None:
        super().__init__()

        self.model = model
        self.n_classes = n_classes
        self.loss = loss
        self.val_confusion = ConfusionMatrix(num_classes=n_classes)#self._config.n_clusters)
        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1Score(number_classes=n_classes,
        average="micro")
        self.train_auroc = torchmetrics.AUROC(num_classes=n_classes,
        average="micro")
        self.val_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1Score(number_classes=n_classes,
        average="micro")
        self.val_auroc = torchmetrics.AUROC(num_classes=n_classes,
        average="micro")
    
    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        loss, pred_labels, true_labels, y, y_hat = self._shared_eval_step(batch, batch_idx)
        self._update_metrics(train=True, ytrue=true_labels, ypred=pred_labels)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        loss, pred_labels, true_labels, y, y_hat = self._shared_eval_step(batch, batch_idx)
        self._update_metrics(train=False, ytrue=true_labels, ypred=pred_labels)
        return {"val_loss": loss}

    def _shared_eval_step(self, batch, batchidx):
        x, z, y = batch
        y_hat = self.model(x,z)
        loss = self.loss(y_hat, y)
        pred_labels = torch.argmax(y_hat, axis=1)
        true_labels = torch.argmax(y, axis=1)
        return loss, pred_labels, true_labels, y_hat, y

    def _update_metrics(self, train: bool, ytrue, ypred):
        if train:
            self.train_acc.update(ypred, ytrue)
            self.train_f1.update(ypred, ytrue)
        else:
            self.val_acc.update(ypred, ytrue)
            self.val_f1.update(ypred, ytrue)
            self.val_confusion(ypred, ytrue)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = self.val_acc.compute()
        avg_f1 = self.val_f1.compute()
        conf_matrix = self.val_confusion.compute()
        # reset metrics
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_confusion.reset()
        # log metrics\
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", avg_acc, prog_bar=True)
        self.log("val_f1", avg_f1, prog_bar=True)
        self.log('confusion_matrix', conf_matrix)

        df_cm = pd.DataFrame(conf_matrix.cpu().numpy(), index = range(self.n_classes), columns=range(self.n_classes))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        
        wandb.log({'conf' :wandb.Image(fig_, caption="Confusion Matrix")})


        return {"val_loss": avg_loss, "val_acc": avg_acc, "val_f1": avg_f1, "confusion_matrix": conf_matrix}


    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = self.train_acc.compute()
        avg_f1 = self.train_f1.compute()
        # reset metrics
        self.train_acc.reset()
        self.train_f1.reset()
        # log metrics
        self.log("train_loss", avg_loss, prog_bar=True)
        self.log("train_acc", avg_acc, prog_bar=True)
        self.log("train_f1", avg_f1, prog_bar=True)
        return {"train_loss": avg_loss, "train_acc": avg_acc, "train_f1": avg_f1}

    def test_step(self, batch, batch_idx, optimizer_idx = 0):
        x, z, y = batch
        y_hat = self.model(x,z )
        loss = self.loss(y_hat, y)
        return {"loss": loss, "pred": y_hat} 
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)