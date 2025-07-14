from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import open_clip
import torch
from torchvision.transforms import v2
from eva.vision.data import datasets
from eva.vision.data.transforms.common import ResizeAndCrop
from sklearn.model_selection import train_test_split
import numpy as np

class CRCDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, download = False, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], size = 224):
        super().__init__()
        self.training_data = datasets.CRC(
                                root="/export/datasets/public/crc",
                                split="train",
                                download = download,
                                transforms = ResizeAndCrop(size = size, mean = mean, std = std),
                            )
        self.training_data.prepare_data()
        self.training_data.configure()
        
        self.test_data = datasets.CRC(
                                root="/export/datasets/public/crc",
                                split="val",
                                download = download,
                                transforms = ResizeAndCrop(size = size, mean = mean, std = std),
                            )
        self.test_data.prepare_data()
        self.test_data.configure()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)