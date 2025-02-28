from Models.TextNetC import TextNetModel
from Models.UNINet import UNINetModel

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
from peft import LoraConfig, get_peft_model
from torch.nn import MultiheadAttention
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from conch.open_clip_custom.transformer import TextTransformer, Transformer, ResidualAttentionBlock
class FuserModel(pl.LightningModule):
    def __init__(self, num_classes = 2, batch_size = 64, lr=0.001, momentum=0.9, nesterov = True, weight_decay = 0.0001, bbfroze = True, lora_r = 16, lora_alpha = 8, text_model = TextNetModel, text_model_path = "./best_models/pcam_TCGA.ckpt", vision_model = UNINetModel, vision_model_path = "./best_models/pcam_UNI.ckpt", scheduler = False, T_max = 12500, eta_min = 0.0001, targets = (nn.Linear, nn.Embedding, nn.Conv2d), lora_text = True, lora_vision = True, lora_dropout = 0.0):#
        super().__init__()
        self.num_classes = num_classes        
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.T_max = T_max
        self.eta_min = eta_min
        self.scheduler = scheduler


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

        self.text_model = text_model(num_classes, batch_size, momentum, nesterov, weight_decay, bbfroze = True)
        checkpoint = torch.load(text_model_path)
        self.text_model.load_state_dict(checkpoint['state_dict'])
                
        if lora_text:
            target_modules = []
            for name, module in self.text_model.named_modules():
                if isinstance(module, targets):#transformers.pytorch_utils.Conv1D
                    target_modules.append(name)
            
            lora_config = LoraConfig(target_modules=target_modules,r=lora_r, lora_alpha = lora_alpha, lora_dropout = lora_dropout)
            self.text_model = get_peft_model(self.text_model, lora_config)

        self.vision_model = vision_model(num_classes, batch_size, lr, momentum, nesterov, weight_decay, bbfroze = True)
        checkpoint = torch.load(vision_model_path)
        state_dict = checkpoint['state_dict']
        if "UNI" in vision_model_path:
            new_state_dict = {}
            for key in state_dict.keys():
                if key == "head.bias":
                    new_key = "fc.bias"
                elif key == "head.weight":
                    new_key = "fc.weight"
                else:
                    new_key = key.replace("backbone._", "").replace("model._", "")
                new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict = {}
            for key in state_dict.keys():
                if key == "head.bias":
                    new_key = "fc.bias"
                elif key == "head.weight":
                    new_key = "fc.weight"
                else:
                    new_key = key.replace("backbone._", "")#.replace("model._", "")
                new_state_dict[new_key] = state_dict[key]
        self.vision_model.load_state_dict(new_state_dict)

        if lora_vision:
            target_modules = []
            for name, module in self.vision_model.named_modules():
                if isinstance(module, targets):
                    target_modules.append(name)
            
            lora_config = LoraConfig(target_modules=target_modules,r=lora_r, lora_alpha = lora_alpha, lora_dropout = lora_dropout)
            self.vision_model = get_peft_model(self.vision_model, lora_config)
        
        
        self.fc = nn.Linear(2 * num_classes, num_classes)
        self.vision_to_text = nn.Linear(self.vision_model.image_embed_size, self.text_model.text_embed_size)
        self.fc0 = nn.Sequential(
                    nn.Linear(self.text_model.text_embed_size, self.text_model.text_embed_size),
                    nn.LeakyReLU(0.1),
                    nn.Linear(self.text_model.text_embed_size, num_classes)
                    )
        print("model created")
        print(self.device)
        

    
    
    def forward(self, x, text_inputs, residual = True):
        self.vision_out, vision_emb = self.vision_model(x)
        vision_to_text = self.vision_to_text(vision_emb)
        self.text_out = self.fc0(vision_to_text)
        x = torch.cat((self.vision_out, self.text_out), 1)
        out = self.fc(x)
        return out, None , vision_emb, vision_to_text
    
    def compute_loss(self, y, yp):
        return F.cross_entropy(y, yp)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        if self.scheduler:
            scheduler = CosineAnnealingLR(optimizer, T_max = self.T_max, eta_min = self.eta_min)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}

    
    def process_batch(self, batch):
        img, txt, lab = batch
        txt = torch.tensor(txt).to(self.device)
        lab = lab.to(self.device)
        out, text_emb, vision_emb, vision_to_text = self.forward(img, txt)
        prd = torch.softmax(out, dim=1)
        loss = self.compute_loss(prd, lab)
        #embedding_a = F.normalize(text_emb, p=2, dim=-1)
        #embedding_b = F.normalize(vision_to_text, p=2, dim=-1)

        #cosine_similarity = torch.sum(embedding_a * embedding_b, dim=-1)
        #loss = loss + (1 - cosine_similarity).mean()
        return loss, prd, lab, text_emb, vision_emb

    def training_step(self, batch, batch_idx):
        loss, prd, lab, _, _ = self.process_batch(batch)
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
        loss, prd, lab, _, _ = self.process_batch(batch)
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
        _, prd, lab, text_emb, vision_emb = self.process_batch(batch)       
        self.predictions.append(prd)
        self.targets.append(lab.squeeze())

