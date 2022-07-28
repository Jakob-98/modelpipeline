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

class config:
    checkpoint_path = "./trained_models/ghostnet_incl_histlbp_54epochs.ckpt"
    # image_path = r"/home/serlierj/datasets/ENA/images/ENA224xCropRGBTrain5/"
    # label_path = r"/home/serlierj/datasets/ENA/labels/ENA224xCropRGBTrain5/"
    # histlbp_path = r"/home/serlierj/datasets/ENA/histlbp/ENA224xCropRGBTrain5/"
    # val_image_path = r"/home/serlierj/datasets/ENA/images/ENA224xCropRGBVal/"
    # val_label_path = r"/home/serlierj/datasets/ENA/labels/ENA224xCropRGBVal/"
    # val_histlbp_path = r"/home/serlierj/datasets/ENA/histlbp/ENA224xCropRGBVal/"
    image_path = r"c:/temp/data_final/islands/images/ISL224xSeqRGBTrain5/"
    label_path = r"c:/temp/data_final/islands/labels/ISL224xSeqRGBTrain5/"
    histlbp_path = r"c:/temp/data_final/islands/histlbp/ISL224xSeqRGBTrain5/"
    val_image_path = r"c:/temp/data_final/islands/images/ISL224xSeqRGBVal20/"
    val_label_path = r"c:/temp/data_final/islands/labels/ISL224xSeqRGBVal20/"
    val_histlbp_path = r"c:/temp/data_final/islands/histlbp/ISL224xSeqRGBVal20/"
    image_size = 224
    nclass = 6

wandb_logger = WandbLogger()
datamodule = dataloaders.DataModuleCustom(
    trainhistlbppath=config.histlbp_path, trainimagepath=config.image_path, 
    trainlabelpath=config.label_path, valhistlbppath=config.val_histlbp_path, valimagepath=config.val_image_path, vallabelpath=config.val_label_path ,nclass=config.nclass)

# %%
# model = ghostnet.ghostnet(num_classes = config.nclass, enable_histlbp=True)
# model.load_state_dict(torch.load('./model_56epochs_inclhist.pt'))
model.eval()
loss = nn.CrossEntropyLoss()
ex = experiment.Experiment(model, loss, config.nclass)
callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =('./checkpoints'), 
                                     monitor= "val_loss",
                                     save_weights_only=True,
                                     save_last= True),
                 ]
trainer = pl.Trainer(callbacks=callbacks, max_epochs=100, accelerator='gpu', logger=wandb_logger)
# %%
datamodule.setup()
trainer.fit(ex, train_dataloaders=datamodule)
# torch.save(model.state_dict(), './ghost_model.pt')
# %%
