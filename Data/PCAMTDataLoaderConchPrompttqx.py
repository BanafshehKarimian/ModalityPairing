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
from conch.open_clip_custom import tokenize, get_tokenizer
from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize, get_tokenizer
import re
checkpoint_path = './CONCH/checkpoints/conch/pytorch_model.bin'
device = "cuda" if torch.cuda.is_available() else "cpu"
class PachDataset(Dataset):

    def __init__(self, split, tqxtype = 1):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.model, self.preprocess_val = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        self.conch_tokenizer = get_tokenizer()
        transform = self.preprocess_val#v2.Compose([ToTensor(),])
        if split == "train":
            transform = v2.Compose([
                self.preprocess_val.transforms[0],
                self.preprocess_val.transforms[1],
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.05),
                self.preprocess_val.transforms[2],
                self.preprocess_val.transforms[3],
                self.preprocess_val.transforms[4],])
        self.data =  datasets.PCAM(
                root="/export/datasets/public/",
                download=True,
                transform= transform,
                split = split
            )
        entity_name = pd.read_csv('text/entity.csv')
        filter_semantic_name = []
        if tqxtype ==1:
            filter_semantic_name = ['Neoplastic Process;']
        elif tqxtype == 2:
            filter_semantic_name = ['Disease or Syndrome;']
        elif tqxtype == 3:
            filter_semantic_name = ['Pathologic Function;']
        if len(filter_semantic_name) != 0:
            mask = entity_name['semantic_name'].isin(filter_semantic_name)
            indices = entity_name.index[mask]
            entity_name = entity_name.iloc[indices].reset_index(drop=True)
        texts = entity_name['entity_name'].tolist()
        self.texts = texts
        self.text_ids = np.load("./Paired_indexes/"+split+"_pcam_indexes_tqx_"+str(tqxtype)+".npy")#self.get_texts()
        
    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, label = self.data.__getitem__(idx)
        txt = tokenize(texts=[self.texts[self.text_ids[idx]]], tokenizer=self.conch_tokenizer)
        return (img, txt, label)

    
class PCAMTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, level = 0):
        super().__init__()
        
        self.training_data = PachDataset("train", level)
        self.test_data = PachDataset("test", level)
        self.valid_data = PachDataset("val", level)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
