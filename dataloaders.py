from importlib.resources import path
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from sklearn import metrics
import time
import copy
import os
import random
from matplotlib import pyplot as plt
import glob 
from pathlib import Path
import PIL
import math
import sys
import torchvision.transforms as transforms
from collections import Counter
from abc import abstractmethod
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Sequence, Union, Any, Callable
import torch.nn.functional as F


import yaml
import argparse
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin
import pytorch_lightning as pl
import torchvision.utils as vutils
from torch import optim



class DatasetLoader(Dataset):
    def __init__(self, imagepath, labelpath, histlbppath, nclass):
        self.imagepath = imagepath
        self.labelpath = labelpath
        self.histlbppath = histlbppath
        self.nclass = nclass 
        self.imageids = sorted([os.path.basename(p).split('.jpg')[0] for p in glob.glob(imagepath + '/*.jpg')])

    
    def one_hot(self, x):
        res = np.zeros(self.nclass)
        res[x-1] = 1
        return res

    def __getitem__(self, index):
        imageId = self.imageids[index]
        histlbp = np.load(Path(self.histlbppath) / (imageId + '.npy'), allow_pickle=False)
        with open(self.labelpath + imageId + '.txt') as f:
            target = f.readline()[0]
        target=self.one_hot(int(target))
        img = np.array(Image.open(os.path.join(self.imagepath, imageId + '.jpg')).convert('RGB'))
        img, target, histlbp = torch.Tensor(img).permute(2,0,1), torch.Tensor(target), torch.Tensor(histlbp)
        # return img, target
        return img, histlbp, target 


    def __len__(self):
        return len(self.imageids)


class DataModuleCustom(pl.LightningDataModule):
    def __init__(
        self,
        nclass,
        trainhistlbppath,
        trainlabelpath,
        trainimagepath,
        valhistlbppath,
        vallabelpath,
        valimagepath
    ):
        super().__init__()
        self.train_batch_size = 16
        self.num_workers = 1
        self.nclass = nclass
        self.trainhistlbppath = trainhistlbppath
        self.trainlabelpath = trainlabelpath
        self.trainimagepath = trainimagepath
        self.valhistlbppath = valhistlbppath
        self.vallabelpath = vallabelpath
        self.valimagepath = valimagepath

    def setup(self, stage: Optional[str] = None) -> None:
        
        self.train_dataset = DatasetLoader(
            histlbppath=self.trainhistlbppath, imagepath=self.trainimagepath, labelpath=self.trainlabelpath, nclass=self.nclass
        )

        self.val_dataset= DatasetLoader(
            histlbppath=self.valhistlbppath, imagepath=self.valimagepath, labelpath=self.vallabelpath, nclass=self.nclass
        )

        self.test_dataset= DatasetLoader(
            histlbppath=self.valhistlbppath, imagepath=self.valimagepath, labelpath=self.vallabelpath, nclass=self.nclass
        )


        # self.val_dataset = ENA(
        # )

        # self.test_dataset = ENA(
        # )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            # num_workers=self.num_workers,
            # shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size = self.train_batch_size
        )
    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.train_batch_size
        )


    # def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.val_batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=self.pin_memory,
    #     )
    
    # def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=144,
    #         num_workers=self.num_workers,
    #         shuffle=True,
    #         pin_memory=self.pin_memory,
    #     )
     