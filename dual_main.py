# %%
import traceback
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch.nn as nn
import dual_dataloaders
# from ghostnet import ghostnet
import dual_ghostnet
from importlib import reload
import experiment
reload(dual_dataloaders)
reload(dual_ghostnet)
reload(experiment)
# from experiment import Experiment


class config:
    image_path1 = "C:/temp/ispipeline/images/224xCropRGBTrain5/"
    label_path = "C:/temp/ispipeline/labels/224xCropRGBTrain5/"
    histlbp_path = "C:/temp/ispipeline/histlbp/224xCropRGBTrain5/"
    val_image_path1 = "C:/temp/ispipeline/images/224xCropRGBval20"
    val_label_path = "C:/temp/ispipeline/labels/224xCropRGBval20/"
    val_histlbp_path = "C:/temp/ispipeline/histlbp/224xCropRGBval20/"
    image_path2 = "C:/temp/ispipeline/images/224xSeqRGBTrain5/"
    # label_path = "C:/temp/ispipeline/labels/224xSeqRGBTrain5/"
    # histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBTrain5/"
    val_image_path2 = "C:/temp/ispipeline/images/224xSeqRGBval20"
    # val_label_path = "C:/temp/ispipeline/labels/224xSeqRGBval20/"
    # val_histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBval20/"
    image_size = 224
    nclass = 6


wandb_logger = WandbLogger()
datamodule = dual_dataloaders.DataModuleCustom(
    trainhistlbppath=config.histlbp_path, trainimagepath2=config.image_path2,
    trainimagepath1=config.image_path1, trainlabelpath=config.label_path,
    valhistlbppath=config.val_histlbp_path, valimagepath1=config.val_image_path1,
    valimagepath2=config.val_image_path2, vallabelpath=config.val_label_path,
    nclass=config.nclass)

# %%
model = dual_ghostnet.DualGhostNet(num_classes=config.nclass)
model.load_state_dict(torch.load('./checkpoints/ghostnet_test.pth'))
# model.eval()
loss = nn.CrossEntropyLoss()
ex = experiment.Experiment(model, loss, n_classes=config.nclass, dual_images=True)
trainer = pl.Trainer(max_epochs=10, accelerator='gpu', logger=wandb_logger)
# %%
try:
    datamodule.setup()
    trainer.fit(ex, train_dataloaders=datamodule)
except Exception as err:
    print(traceback.format_exc())

# %%
torch.save(model.state_dict(), './modeldual_12epochs.pt')
# %%%

trainer.validate(ex, datamodule=datamodule, verbose=True)


# USELESS
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
