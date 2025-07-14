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
from peft import LoraConfig, get_peft_model
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
from conch.open_clip_custom import tokenize, get_tokenizer
from conch.open_clip_custom import create_model_from_pretrained
checkpoint_path = '/export/livia/home/vision/Bkarimian/CONCH/checkpoints/conch/pytorch_model.bin'

class CONCHNetModel(pl.LightningModule):
    def __init__(self, class_names = [], num_classes = 2, batch_size = 64, lr=0.001, momentum=0.9, nesterov = True, weight_decay = 0.0001, bbfroze = True, lora_r = 16, lora_alpha = 8, targets = (nn.Linear, nn.Embedding, nn.Conv2d)):
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
            self.metric = MulticlassAccuracy(num_classes=num_classes).to(self.device)

        self.model, self.preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path)
        for param in self.model.parameters():
            param.requires_grad = False
        target_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, targets):
                target_modules.append(name)
        
        lora_config = LoraConfig(target_modules=target_modules,r=lora_r, lora_alpha = lora_alpha)
        self.model = get_peft_model(self.model, lora_config)
        #_ = self.model.eval()
        self.text_embed_size = 512
        self.class_names = class_names
        self.tokenizer = get_tokenizer()
        self.templates = [
                    "an H&E image of CLASSNAME.",
                    "CLASSNAME.",
                    "a photomicrograph showing CLASSNAME.",
                    "a photomicrograph of CLASSNAME.",
                    "an image of CLASSNAME.",
                    "an image showing CLASSNAME.",
                    "an example of CLASSNAME.",
                    "CLASSNAME is shown.",
                    "this is CLASSNAME.",
                    "there is CLASSNAME.",
                    "a histopathological image showing CLASSNAME.",
                    "a histopathological image of CLASSNAME.",
                    "a histopathological photograph of CLASSNAME.",
                    "a histopathological photograph showing CLASSNAME.",
                    "shows CLASSNAME.",
                    "presence of CLASSNAME.",
                    "CLASSNAME is present.",
                    "an H&E stained image of CLASSNAME.",
                    "an H&E stained image showing CLASSNAME.",
                    "an H&E image showing CLASSNAME.",
                    "CLASSNAME, H&E stain.",
                    "CLASSNAME, H&E."]
        self.k_prompts_per_class = len(self.templates)
        self.tokenized_prompts = self.get_class_text_embedding(self.templates)
        print("model created")
        print(self.device)

            
    def get_class_text_embedding(self, templates):
        prompts = []
        for class_name in self.class_names:
            for tmp in templates:
                prompts.append(tmp.replace("CLASSNAME", class_name))
        tokenized_prompts = tokenize(texts=prompts, tokenizer=self.tokenizer).to(self.device)
        return tokenized_prompts

    
    def compute_loss(self, image_features, text_features, temperature=0.07):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits_per_image = image_features @ text_features.T  # [B, B]
        logits_per_text = text_features @ image_features.T  # [B, B]

        logits_per_image = logits_per_image / temperature
        logits_per_text = logits_per_text / temperature

        # Labels are positions along the diagonal
        labels = torch.arange(image_features.size(0), device=image_features.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2

    def configure_optimizers(self):
        #optimizer = optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum, nesterov = self.nesterov, weight_decay = self.weight_decay)
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        #scheduler = CosineAnnealingLR(optimizer, T_max = 12500, eta_min = 0.0001)
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler

    def forward(self, img):
        
        self.model.to(self.device)
        k = self.k_prompts_per_class
        tokenized_prompts = self.get_class_text_embedding(self.templates)
        with torch.inference_mode():
            image_embedings = self.model.encode_image(img)
            text_embedings = self.model.encode_text(tokenized_prompts.to(self.device))
            sim_scores = (image_embedings @ text_embedings.T * self.model.logit_scale.exp())#.cpu()#.numpy()    
            num_classes = sim_scores.shape[1] // k
            sim_scores = sim_scores.view(sim_scores.shape[0], num_classes, k)
            sim_scores = sim_scores.mean(dim=2) #mean?
            prd = sim_scores.softmax(dim=-1)
    
    def process_batch(self, batch, use_prompt = True):
        img, txt, lab = batch
        self.model.to(self.device)
        k = self.k_prompts_per_class
        image_embedings = self.model.encode_image(img)
        tokenized_prompts_train = self.get_class_text_embedding(self.templates[:1])
        if use_prompt:
            text = torch.stack([tokenized_prompts_train]*image_embedings.shape[0]).to(self.device)
            text = text[torch.arange(text.shape[0]), lab]
        else:
            text = txt.squeeze(dim=1)
        text_embedings = self.model.encode_text(text)
        loss = self.compute_loss(image_embedings, text_embedings)
        text_embedings = self.model.encode_text(tokenized_prompts_train.to(self.device))
        prd = (image_embedings @ text_embedings.T * self.model.logit_scale.exp()).softmax(dim=-1)
        return loss, prd, lab

    def training_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.train_step_preds.append(prd.cpu().detach())
        self.train_step_trgts.append(lab.cpu().detach())
        self.log('train_loss', loss, batch_size=self.batch_size)        
        return loss

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_step_preds, dim=0).to(self.device)
        all_trgts = torch.cat(self.train_step_trgts, dim=0).to(self.device)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('train_auc', auc, batch_size=len(all_preds))
        acc = self.metric(all_preds.argmax(1), all_trgts)
        self.log('train_acc', acc, batch_size=len(all_preds))
        self.train_step_preds.clear()
        self.train_step_trgts.clear()

    def validation_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.val_step_preds.append(prd.cpu().detach())
        self.val_step_trgts.append(lab.cpu().detach())
        self.log('val_loss', loss, batch_size=self.batch_size)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_preds, dim=0).to(self.device)
        all_trgts = torch.cat(self.val_step_trgts, dim=0).to(self.device)
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
        #loss, prd, lab = self.process_batch(batch)
        img, txt, lab = batch
        self.model.to(self.device)
        k = self.k_prompts_per_class
        tokenized_prompts = self.get_class_text_embedding(self.templates)
        with torch.inference_mode():
            image_embedings = self.model.encode_image(img)
            text_embedings = self.model.encode_text(tokenized_prompts.to(self.device))
            sim_scores = (image_embedings @ text_embedings.T * self.model.logit_scale.exp())#.cpu()#.numpy()    
            num_classes = sim_scores.shape[1] // k
            sim_scores = sim_scores.view(sim_scores.shape[0], num_classes, k)
            sim_scores = sim_scores.mean(dim=2) #mean?
            prd = sim_scores.softmax(dim=-1)
        self.predictions.append(prd)
        self.targets.append(lab.squeeze())

        