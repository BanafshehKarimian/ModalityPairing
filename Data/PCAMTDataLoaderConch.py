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
from utils.utils import *
import torch.nn as nn
# Load model directly
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

processor = AutoProcessor.from_pretrained("vinid/plip")
checkpoint_path = '/export/livia/home/vision/Bkarimian/CONCH/checkpoints/conch/pytorch_model.bin'
device = "cuda" if torch.cuda.is_available() else "cpu"
def resize_token_embeddings(model, tokenizer):
    tokenizer.add_special_tokens({'mask_token': '<mask>'})
    mask_id = tokenizer.mask_token_id
    embed_layer = model.text.token_embedding
    old_weights = embed_layer.weight.data
    old_num, dim = old_weights.shape
    new_num = len(tokenizer)
    new_embed = nn.Embedding(new_num, dim)
    new_embed.weight.data[:old_num] = old_weights
    eos_id = tokenizer.eos_token_id
    new_embed.weight.data[old_num:] = old_weights[eos_id].unsqueeze(0)
    model.text.token_embedding = new_embed
    return model, tokenizer, mask_id

def mask_k_percent(input_ids: torch.LongTensor,
                   mask_id: int,
                   k: float = 0.1,
                   eos_id: int = 2) -> torch.LongTensor:
    seq = input_ids.clone()
    _, N = seq.shape

    eos_positions = torch.nonzero(seq == eos_id, as_tuple=False)
    if eos_positions.numel() > 0:
        end_idx = int(eos_positions[0, 1])
    else:
        end_idx = N

    num_to_mask = max(1, int(end_idx * k))

    perm = torch.randperm(end_idx)
    mask_indices = perm[:num_to_mask]
    seq[0, mask_indices] = mask_id

    return seq
    
class PachDataset(Dataset):

    def __init__(self, split, transformer, raw_text, mask = False, no_text=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mask = mask
        self.conch_tokenizer = get_tokenizer()
        if transformer in ["CONCH"]:
            self.model, self.preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path)
            self.processor = AutoProcessor.from_pretrained("vinid/plip")
            #if split == "train":
            transform = self.preprocess
            if mask:
                self.model, self.conch_tokenizer, self.mask_token_id = resize_token_embeddings(self.model, self.conch_tokenizer)
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
        self.raw_text = raw_text
        self.data =  datasets.PCAM(
                root="/export/datasets/public/",
                download=True,
                transform= transform,
                split = split
            )
        with open("text/text_breast.txt") as file:
            self.texts = [clean_pathology_report(line.rstrip()) for line in file]
        self.text_ids = np.load("./Paired_indexes/pcam_"+split+"_indexes.npy")
        self.no_text = no_text
        
    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, label = self.data.__getitem__(idx)
        if self.raw_text:
            return (img, self.texts[self.text_ids[idx]], label)
        txt = tokenize(texts=[self.texts[self.text_ids[idx]]], tokenizer=self.conch_tokenizer)#self.processor(self.texts[self.text_ids[idx]],padding=True,truncation=True, max_length=77,return_tensors="pt")['input_ids']#
        if self.mask:
            txt = mask_k_percent(txt, self.mask_token_id)
        if self.no_text:
            return (img, label)
        return (img, txt, label)

    
class PCAMTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, transformer = "CONCH", raw_text = False, no_text=False):
        super().__init__()
        
        self.training_data = PachDataset("train", transformer, raw_text, no_text=no_text)
        self.test_data = PachDataset("test", transformer, raw_text, no_text=no_text)
        self.valid_data = PachDataset("val", transformer, raw_text, no_text=no_text)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)