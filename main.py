# %%
import dataloaders
from ghostnet import ghostnet
from importlib import reload
reload(dataloaders)
from experiment import Experiment
import torch.nn as nn
import pytorch_lightning as pl


class config:
    image_path = "C:/temp/ispipeline/images/224xSeqRGBTrain5/"
    label_path = "C:/temp/ispipeline/labels/224xSeqRGBTrain5/"
    histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBTrain5/"
    image_size = 224
    nclass = 6


datamodule = dataloaders.DataModuleCustom(
    trainhistlbppath=config.histlbp_path, trainimagepath=config.image_path, 
    trainlabelpath=config.label_path, nclass=config.nclass)

# %%
model = ghostnet()
loss = nn.CrossEntropyLoss()
experiment = Experiment(model, loss)
trainer = pl.Trainer(max_epochs=1)
# %%
datamodule.setup()
# trainer.fit(experiment, train_dataloaders=datamodule)
# %%
traindl = datamodule.train_dataloader()
# %%
batch = next(iter(traindl))
# %%
