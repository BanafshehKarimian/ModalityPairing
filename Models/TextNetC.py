import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from torch import optim
import open_clip
import pytorch_lightning as pl
import open_clip
from PIL import Image
from torchmetrics.functional import auroc
import timm
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
from conch.open_clip_custom import tokenize, get_tokenizer
from conch.open_clip_custom import create_model_from_pretrained
checkpoint_path = './CONCH/checkpoints/conch/pytorch_model.bin'

class TextNetModel(pl.LightningModule):
    def __init__(self, num_classes = 2, batch_size = 64, lr=0.001, momentum=0.9, nesterov = True, weight_decay = 0.0001, bbfroze = True):
        super().__init__()
        self.num_classes = num_classes        
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.predictions = []
        self.targets = []

        self.train_step_preds = []
        self.train_step_trgts = []
        self.val_step_preds = []
        self.val_step_trgts = []
        self.train_loss = []
        self.val_loss = []

        if num_classes == 2:
            self.metric = BinaryAccuracy()
        else:
            self.metric = MulticlassAccuracy(num_classes=num_classes)

        self.model, self.preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path)
        _ = self.model.eval()
        if bbfroze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.text_embed_size = 512
        self.fc0 = nn.Linear(self.text_embed_size, self.text_embed_size)
        self.m = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(self.text_embed_size, num_classes)
        print("model created")
        print(self.device)
        

    
    def forward(self, img, text_inputs, residual = True):
        x = self.model.encode_text(text_inputs.squeeze(1))
        out = self.fc(self.m(self.fc0(x)))
        return out, x
    
    def compute_loss(self, y, yp):
        return F.cross_entropy(y, yp)

    def configure_optimizers(self):
        #optimizer = optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum, nesterov = self.nesterov, weight_decay = self.weight_decay)
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max = 12500, eta_min = 0.0001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}#

    def process_batch(self, batch):
        img, txt, lab = batch
        txt = torch.tensor(txt).to(self.device)
        lab = lab.to(self.device)
        out, _ = self.forward(img, txt)
        prd = torch.softmax(out, dim=1)
        loss = self.compute_loss(prd, lab)
        return loss, prd, lab

    def training_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.train_step_preds.append(prd)
        self.train_step_trgts.append(lab)
        self.log('train_loss', loss, batch_size=self.batch_size)        
        return loss

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_step_preds, dim=0)
        all_trgts = torch.cat(self.train_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('train_auc', auc, batch_size=len(all_preds))
        acc = self.metric(all_preds.argmax(1), all_trgts)
        self.log('train_acc', acc, batch_size=len(all_preds))
        self.train_step_preds.clear()
        self.train_step_trgts.clear()

    def validation_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.val_step_preds.append(prd)
        self.val_step_trgts.append(lab)
        self.log('val_loss', loss, batch_size=self.batch_size)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_preds, dim=0)
        all_trgts = torch.cat(self.val_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('val_auc', auc, batch_size=len(all_preds))
        acc = self.metric(all_preds.argmax(1), all_trgts)
        self.log('val_acc', acc, batch_size=len(all_preds))
        self.val_step_preds.clear()
        self.val_step_trgts.clear()

    def on_test_start(self):
        self.predictions = []
        self.targets = []

    def test_step(self, batch, batch_idx):
        _, prd, lab = self.process_batch(batch)        
        self.predictions.append(prd)
        self.targets.append(lab.squeeze())

        