from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import open_clip
import torch
from torchvision.transforms import v2
class PCAMDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, download = True, mean = None, std = None):
        super().__init__()
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        transform = v2.Compose([
                self.preprocess_val.transforms[0],
                self.preprocess_val.transforms[1],
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.05),
                self.preprocess_val.transforms[2],
                self.preprocess_val.transforms[3],
                self.preprocess_val.transforms[4],])
        self.training_data = datasets.PCAM(
                root="/export/datasets/public/",
                download=download,
                transform= transform
            )

        self.test_data = datasets.PCAM(
            root="/export/datasets/public/",
            download=download,
            transform=self.preprocess_val,
            split = "test"
        )
        
        self.valid_data = datasets.PCAM(
            root="/export/datasets/public/",
            download=download,
            transform=self.preprocess_val,
            split = "val"
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)