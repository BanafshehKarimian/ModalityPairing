from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import open_clip
import torch
import pandas as pd
import json
from torchvision import transforms
import numpy as np
from torchvision.transforms import v2
from eva.vision.data import datasets
from eva.vision.data.transforms.common import ResizeAndCrop
from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize, get_tokenizer
from sklearn.model_selection import train_test_split
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = './CONCH/checkpoints/conch/pytorch_model.bin'

class CRCDataset(Dataset):

    def __init__(self, split, mean, std):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.model, self.preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
        self.model = self.model.to(device)
        _ = self.model.eval()
        self.conch_tokenizer = get_tokenizer()
        self.data =  datasets.CRC(
                                root="/export/datasets/public/crc",
                                split=split,
                                download = False,
                                transforms = ResizeAndCrop(size = 224, mean = mean, std = std),#, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
                            )
        
        self.data.prepare_data()
        self.data.configure()
        with open("text/text_colo.txt") as file:
            self.texts = [line.rstrip() for line in file]

        self.text_ids = np.load("./Paired_indexes/crc_"+split+"_indexes.npy")#self.get_texts()

    def __len__(self):
        return self.data.__len__()

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, label, _ = self.data.__getitem__(idx)
        txt = tokenize(texts=[self.texts[self.text_ids[idx]]], tokenizer=self.conch_tokenizer)
        '''dot_prod = (self.image_embeddings[idx]*self.embed_tensor).sum(dim = 1)
        txt = self.tokenizer(self.texts[dot_prod.cpu().argmax().item()])'''
        return (img, txt, label)#self.tokenizer()

class CRCTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]):
        super().__init__()
        
        training_data = CRCDataset("train", mean, std)
        self.test_data = CRCDataset("val", mean, std)
        labels = np.load("crc_labels.npy")#[training_data[i][2] for i in range(len(training_data))]
        #train = [i for i in range(len(training_data)) if i%20 != 0]
        #val = [i for i in range(len(training_data)) if i%20 == 0]
        train, val = train_test_split(np.arange(len(training_data)),
                                             test_size=0.2,
                                             random_state=42,
                                             shuffle=True,
                                             stratify=labels)
        self.training_data = torch.utils.data.Subset(training_data, train)
        self.val_data = torch.utils.data.Subset(training_data, val)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
