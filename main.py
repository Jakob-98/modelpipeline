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
    image_path = r"C:\temp\data_final\ENA\images\ENA224xCropRGBTrain5"
    label_path = r"C:\temp\data_final\ENA\labels\ENA224xCropRGBTrain5"
    histlbp_path = r"C:\temp\data_final\ENA\histlbp\ENA224xCropRGBTrain5"
    val_image_path = r"C:\temp\data_final\ENA\images\ENA224xCropRGBVal"
    val_label_path = r"C:\temp\data_final\ENA\labels\ENA224xCropRGBVal" 
    val_histlbp_path = r"C:\temp\data_final\ENA\histlbp\ENA224xCropRGBVal"
    # image_path = "C:/temp/ispipeline/images/224xCropRGBTrain5/"
    # label_path = "C:/temp/ispipeline/labels/224xCropRGBTrain5/"
    # histlbp_path = "C:/temp/ispipeline/histlbp/224xCropRGBTrain5/"
    # val_image_path = "C:/temp/ispipeline/images/224xCropRGBval20"
    # val_label_path = "C:/temp/ispipeline/labels/224xCropRGBval20/"
    # val_histlbp_path = "C:/temp/ispipeline/histlbp/224xCropRGBval20/"
    # image_path = "C:/temp/ispipeline/images/224xSeqRGBTrain5/"
    # label_path = "C:/temp/ispipeline/labels/224xSeqRGBTrain5/"
    # histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBTrain5/"
    # val_image_path = "C:/temp/ispipeline/images/224xSeqRGBval20"
    # val_label_path = "C:/temp/ispipeline/labels/224xSeqRGBval20/"
    # val_histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBval20/"
    image_size = 224
    nclass = 21

wandb_logger = WandbLogger()
datamodule = dataloaders.DataModuleCustom(
    trainhistlbppath=config.histlbp_path, trainimagepath=config.image_path, 
    trainlabelpath=config.label_path, valhistlbppath=config.val_histlbp_path, valimagepath=config.val_image_path, vallabelpath=config.val_label_path ,nclass=config.nclass)

# %%
model = ghostnet.ghostnet(num_classes = config.nclass, enable_histlbp=True)
# model.load_state_dict(torch.load('./model.pt'))
model.eval()
loss = nn.CrossEntropyLoss()
ex = experiment.Experiment(model, loss, config.nclass)
trainer = pl.Trainer(max_epochs=300, accelerator='gpu', logger=wandb_logger)
# %%
datamodule.setup()
trainer.fit(ex, train_dataloaders=datamodule)
torch.save(model.state_dict(), './ghost_model.pt')
# %%
