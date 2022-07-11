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

class Experiment(pl.LightningModule):
    def __init__(self, model, loss) -> None:
        super().__init__()

        self.model = model
        self.loss = loss
        self.val_confusion = ConfusionMatrix(num_classes=6)#self._config.n_clusters)
    
    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        loss, pred_labels, true_labels, y, y_hat = self._shared_eval_step(batch, batch_idx)
        acc = accuracy(pred_labels, true_labels)
        metrics = {"loss": loss, 'acc': acc} 
        self.log_dict(metrics)
        return metrics
    
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        loss, pred_labels, true_labels, y, y_hat = self._shared_eval_step(batch, batch_idx)
        acc = accuracy(pred_labels, true_labels)
        metrics = {"val_loss": loss, 'val_acc': acc} 
        self.log_dict(metrics)
        # self.val_confusion.update(pred_labels, true_labels)
        return metrics

    def _shared_eval_step(self, batch, batchidx):
        x, z, y = batch
        y_hat = self.model(x,z)
        loss = self.loss(y_hat, y)
        pred_labels = torch.argmax(y_hat, axis=1)
        true_labels = torch.argmax(y, axis=1)
        return loss, pred_labels, true_labels, y_hat, y


    # def validation_step_end(self, outputs):
    #     return outputs

    # def validation_epoch_end(self, outs):
    #     # https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning
    #     tb = self.logger.experiment

    #     # confusion matrix
    #     conf_mat = self.val_confusion.compute().detach().cpu().numpy().astype(np.int)
    #     df_cm = pd.DataFrame(
    #         conf_mat,
    #         index=np.arange(6),#self._config.n_clusters),
    #         columns=np.arange(6))#self._config.n_clusters))
    #     plt.figure()
    #     sn.set(font_scale=1.2)
    #     sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
    #     buf = io.BytesIO()
        
    #     plt.savefig(buf, format='jpeg')
    #     buf.seek(0)
    #     im = Image.open(buf)
    #     im = torchvision.transforms.ToTensor()(im)
    #     tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)

    def test_step(self, batch, batch_idx, optimizer_idx = 0):
        x, z, y = batch
        y_hat = self.model(x,z )
        loss = self.loss(y_hat, y)
        return {"loss": loss, "pred": y_hat} 
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)