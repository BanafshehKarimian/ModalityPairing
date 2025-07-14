from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import open_clip
import torch
from torchvision.transforms import v2
from conch.open_clip_custom import create_model_from_pretrained
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

checkpoint_path = '/export/livia/home/vision/Bkarimian/CONCH/checkpoints/conch/pytorch_model.bin'
class PCAMDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, download = True, mean = None, std = None, transformer = "CONCH"):
        super().__init__()
        
        if transformer in ["CONCH"]:
            self.model, self.preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path)
            self.processor = AutoProcessor.from_pretrained("vinid/plip")
            #if split == "train":
            transform = self.preprocess
        elif transformer in ["QUILT"]:
            self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
            transform = v2.Compose([
                    self.preprocess_val.transforms[0],
                    self.preprocess_val.transforms[1],
                    #v2.RandomHorizontalFlip(p=0.5),
                    #v2.RandomVerticalFlip(p=0.5),
                    #v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.05),
                    self.preprocess_val.transforms[2],
                    self.preprocess_val.transforms[3],
                    self.preprocess_val.transforms[4],])
        else:
            transform = v2.Compose([
                            v2.Resize(224),
                            v2.CenterCrop(224),
                            v2.ToTensor(),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
                        ])
        self.training_data = datasets.PCAM(
                root="/export/datasets/public/",
                download=download,
                transform= transform
            )

        self.test_data = datasets.PCAM(
            root="/export/datasets/public/",
            download=download,
            transform=transform,
            split = "test"
        )
        
        self.valid_data = datasets.PCAM(
            root="/export/datasets/public/",
            download=download,
            transform=transform,
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