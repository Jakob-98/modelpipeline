# %%
import dataloaders
# from ghostnet import ghostnet
import ghostnet
from importlib import reload
import experiment
reload(dataloaders)
reload(ghostnet)
reload(experiment)
# from experiment import Experiment
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

class config:
    image_path = "C:/temp/ispipeline/images/224xSeqRGBTrain5/"
    label_path = "C:/temp/ispipeline/labels/224xSeqRGBTrain5/"
    histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBTrain5/"
    val_image_path = "C:/temp/ispipeline/images/224xSeqRGBval20/"
    val_label_path = "C:/temp/ispipeline/labels/224xSeqRGBval20/"
    val_histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBval20/"
    image_size = 224
    nclass = 6

wandb_logger = WandbLogger()
datamodule = dataloaders.DataModuleCustom(
    trainhistlbppath=config.histlbp_path, trainimagepath=config.image_path, 
    trainlabelpath=config.label_path, valhistlbppath=config.val_histlbp_path, valimagepath=config.val_image_path, vallabelpath=config.val_label_path ,nclass=config.nclass)

# %%
model = ghostnet.ghostnet(num_classes = config.nclass)
# torch.save(model.state_dict(), './model.pt')
model.load_state_dict(torch.load('./model.pt'))
model.eval()
loss = nn.CrossEntropyLoss()
ex = experiment.Experiment(model, loss)
trainer = pl.Trainer(max_epochs=3, accelerator='gpu', logger=wandb_logger)
# %%
datamodule.setup()
# trainer.fit(ex, train_dataloaders=datamodule)
trainer.validate(ex, datamodule=datamodule, verbose=True)
# %%
traindl = datamodule.train_dataloader()
# %%
batch = next(iter(traindl))
# %%
dataloader = dataloaders.DatasetLoader(
    histlbppath=config.histlbp_path, imagepath=config.image_path, labelpath=config.label_path, nclass=config.nclass)

x, i, y = dataloader.__getitem__(34)

# %%
model.forward((x[0][None, :], x[1][None, :]))
# %%
