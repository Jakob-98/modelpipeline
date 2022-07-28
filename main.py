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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

########################################################################################################################
# standard argparser for config
########################################################################################################################

import argparse, yaml

parser = argparse.ArgumentParser(description='Configure a jconfig')
parser.add_argument('-c', '--config', help='Config .yaml file path', type=str, default='./config.yaml')
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configyaml = yaml.load(f, Loader=yaml.FullLoader)

class config:
    for key, value in configyaml.items():
        locals()[key] = value

########################################################################################################################
# main

# class config:
#     image_path = r"/home/serlierj/datasets/ENA/images/ENA224xCropRGBTrain5/"
#     label_path = r"/home/serlierj/datasets/ENA/labels/ENA224xCropRGBTrain5/"
#     histlbp_path = r"/home/serlierj/datasets/ENA/histlbp/ENA224xCropRGBTrain5/"
#     val_image_path = r"/home/serlierj/datasets/ENA/images/ENA224xCropRGBVal/"
#     val_label_path = r"/home/serlierj/datasets/ENA/labels/ENA224xCropRGBVal/"
#     val_histlbp_path = r"/home/serlierj/datasets/ENA/histlbp/ENA224xCropRGBVal/"
#     image_size = 224
#     nclass = 6
#     max_epochs = 10
#     enable_histlbp = True

wandb_logger = WandbLogger()
datamodule = dataloaders.DataModuleCustom(
    trainhistlbppath=config.histlbp_path, trainimagepath=config.image_path, 
    trainlabelpath=config.label_path, valhistlbppath=config.val_histlbp_path, valimagepath=config.val_image_path, vallabelpath=config.val_label_path ,nclass=config.nclass)

# %%
model = ghostnet.ghostnet(num_classes = config.nclass, enable_histlbp=config.enable_histlbp)
# model.load_state_dict(torch.load('./model.pt'))
model.eval()
loss = nn.CrossEntropyLoss()
ex = experiment.Experiment(model, loss, config.nclass)
trainer = pl.Trainer(max_epochs=config.max_epochs, accelerator='gpu', logger=wandb_logger)
# %%
datamodule.setup()
trainer.fit(ex, train_dataloaders=datamodule)
torch.save(model.state_dict(), f'./{config.ex_name}_final_model.pt')
# %%
