from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import open_clip
import torch
from torchvision.transforms import v2
from eva.vision.data import datasets
from eva.vision.data.transforms.common import ResizeAndCrop

class BACHDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, download = True, size = 224, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]):
        super().__init__()
        self.training_data = datasets.BACH(
                                root="/export/datasets/public/BACH",
                                split="train",
                                download = download,
                                transforms = ResizeAndCrop(size = size, mean = mean, std = std),
                            )
        self.training_data.prepare_data()
        self.training_data.configure()
        train = [i for i in range(len(self.training_data)) if i%20 != 0]
        val = [i for i in range(len(self.training_data)) if i%20 == 0]
        self.training_data = torch.utils.data.Subset(self.training_data, train)
        self.val_data = torch.utils.data.Subset(self.training_data, val)
        
        self.test_data = datasets.BACH(
                                root="/export/datasets/public/BACH",
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
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)