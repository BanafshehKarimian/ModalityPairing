from Data.PCAMDataLoader import PCAMDataModule
from Data.BACHDataLoader import BACHDataModule
from Data.TCGATextLoader import TCGAText
from Data.CRCDataLoader import CRCDataModule
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import argparse
from sklearn.utils import shuffle
import pytorch_lightning as pl
import torch
import pandas as pd
import open_clip
import torch
import json
from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize, get_tokenizer
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
def get_text_embedding(text, tokenizer, model):
    with torch.no_grad():
        text = tokenize(texts=[text], tokenizer=tokenizer).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True) 
    return text_features.squeeze(0).cpu().numpy().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, default=None)
    parser.add_argument("--save-text", action="store_true")
    parser.add_argument("--text-dir", type=str, default='./text/text_')
    parser.add_argument("--ds", type=str, default="pcam")
    args = parser.parse_args()
    text = TCGAText(args.keyword).report_text
    if args.save_text:
        with open(args.text_dir+args.keyword+'.txt', 'w') as f:
            for line in text:
                f.write(f"{line[0]}\n")
    checkpoint_path = './CONCH/checkpoints/conch/pytorch_model.bin'
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
    model = model.to(device)
    _ = model.eval()
    tokenizer = get_tokenizer()
    text_emb = []
    print("calculating the embeddings:")
    for i in tqdm(range(len(text))):
        text_emb.append(get_text_embedding(text[i][0], tokenizer, model))
    
