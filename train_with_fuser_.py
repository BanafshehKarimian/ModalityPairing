from Data.PCAMTDataLoaderConch import PCAMTDataModule
from Data.BACHTDataLoaderConch import BACHTDataModule
from Data.CRCTDataLoaderConch import CRCTDataModule
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.utils import shuffle
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from Models.LateFuser_ import FuserModel, FuserModel_
from torchmetrics.functional import auroc
import numpy as np
import random
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
import argparse
from utils.utils import *
from Models.TextNetC import TextNetModel
from Models.UNINet import UNINetModel
from Models.DINOL14Net import DINOL14NetModel
from Models.VITSNet import VITSNetModel8, VITSNetModel16
from Models.VITBNet import VITBNetModel8, VITBNetModel16
from lightning.pytorch.loggers import WandbLogger
import wandb
import torch.nn as nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="UNI")
    parser.add_argument("--monitor", type=str, default="val_loss")
    parser.add_argument("--mod", type=str, default="min")
    parser.add_argument("--ds", type=str, default="pcam")
    parser.add_argument("--learner", type=str, default="late_fusion_kd")
    parser.add_argument("--dir", type=str, default='output')
    parser.add_argument("--output", type=str, default='train_with_text')
    parser.add_argument("--run", type=str, default='run1')
    parser.add_argument("--sd", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--delta", type=float, default=5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--val-int", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--log", type=int, default=1)
    parser.add_argument("--early", action="store_true")
    parser.add_argument("--lora-text", type=int, default=1)
    parser.add_argument("--lora-vision", type=int, default=1)
    parser.add_argument("--kd-layers", type=int, default=1)
    parser.add_argument("--scheduler", type=int, default=0)
    parser.add_argument("--no-lin", action="store_true")
    parser.add_argument("--chkpnt", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()


    output_base_dir = args.dir
    output_name = args.output
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.wandb:
        wandb.init(project=args.ds + "_" + args.model, reinit=True)
        wandb.config.update(args)

    
    seed_function(args.sd)
    datasets_conf = {"pcam":{"num_classes": 2, "ds_class": PCAMTDataModule, "text_model": TextNetModel, "text_model_path": "./best_models/pcam_TCGA.ckpt"},\
                    "bach":{"num_classes": 4, "ds_class": BACHTDataModule, "text_model": TextNetModel, "text_model_path": "./best_models/bach_TCGA.ckpt"},\
                    "crc":{"num_classes": 9, "ds_class": CRCTDataModule, "text_model": TextNetModel, "text_model_path": "./best_models/crc_TCGA.ckpt"}}
    model_conf = {"UNI": {"model": UNINetModel, "path": {"pcam": "./best_models/run1/pcam_UNI.ckpt", "bach": "./best_models/run1/bach_UNI.ckpt", "crc": "./best_models/run1/crc_UNI.ckpt"}},\
                    "DINOL14": {"model": DINOL14NetModel, "path": {"pcam": "./best_models/run1/pcam_DINOL14.ckpt", "bach": "./best_models/run1/bach_vitl14.ckpt", "crc": "./best_models/run1/crc_vitl14.ckpt"}},\
                    "VITS_8": {"model": VITSNetModel8, "path": {"pcam": "./best_models/run1/pcam_VITS_8.ckpt", "bach": "./best_models/run1/bach_vits8.ckpt", "crc": "./best_models/run1/crc_vits8.ckpt"}},\
                    "VITS_16": {"model": VITSNetModel16, "path": {"pcam": "./best_models/run1/pcam_VITS_16.ckpt", "bach": "./best_models/run1/bach_vits16.ckpt", "crc": "./best_models/run1/crc_vits16.ckpt"}},\
                    "VITB_8": {"model": VITBNetModel8, "path": {"pcam": "./best_models/run1/pcam_VITB_8.ckpt", "bach": "./best_models/run1/bach_vitb8.ckpt", "crc": "./best_models/run1/crc_vitb8.ckpt"}},\
                    "VITB_16": {"model": VITBNetModel16, "path": {"pcam": "./best_models/run1/pcam_VITB_16.ckpt", "bach": "./best_models/run1/bach_vitb16.ckpt", "crc": "./best_models/run1/crc_vitb16.ckpt"}}}
    learners = {"late_fusion": FuserModel, "late_fusion_kd": FuserModel}

    epochs = args.ep
    device = "gpu" if torch.cuda.is_available() else "cpu"
    
    num_classes = datasets_conf[args.ds]["num_classes"]
    batch_size, num_workers = args.batch, args.worker
    Net = learners[args.learner]
    '''Loading Data'''
    if args.model in ["UNI"] and args.ds in ["bach", "crc", "mhist"]:
        data = datasets_conf[args.ds]["ds_class"](batch_size, num_workers, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    else:
        data = datasets_conf[args.ds]["ds_class"](batch_size, num_workers)
    '''Creating the model'''
    targets = (nn.Linear, nn.Embedding, nn.Conv2d)
    if args.no_lin:
        targets = (nn.Embedding, nn.Conv2d)
    model_conf[args.model]["path"][args.ds] = model_conf[args.model]["path"][args.ds].replace("run1", args.run)
    model = Net(num_classes, batch_size, targets = targets, lr = args.lr, text_model = datasets_conf[args.ds]["text_model"], text_model_path = datasets_conf[args.ds]["text_model_path"], vision_model = model_conf[args.model]["model"], vision_model_path = model_conf[args.model]["path"][args.ds], lora_r = args.lora_r, lora_alpha = args.lora_alpha, lora_text = args.lora_text, lora_vision = args.lora_vision, weight_decay = args.weight_decay, scheduler = args.scheduler, lora_dropout =  args.lora_dropout)

    print('=============================================================')
    print('Training...')
    print(device)

    checkpoint_callback = ModelCheckpoint(monitor=args.monitor, mode=args.mod)
    early_stop_callback = EarlyStopping(
            monitor="train_loss",
            min_delta=args.delta,
            patience=args.patience,  # NOTE no. val epochs, not train epochs
            verbose=False,
            mode="min",
        )
    callbacks=[]
    if args.chkpnt:
        callbacks=[checkpoint_callback]
    if args.early:
        callbacks.append(early_stop_callback)
    logger=TensorBoardLogger(output_base_dir, name=output_name)
    if args.wandb:
        wandb_logger = WandbLogger(log_model=False)
        logger = [wandb_logger, logger]
    trainer = pl.Trainer(
            callbacks = callbacks,
            log_every_n_steps=args.log,
            max_epochs=epochs,
            accelerator=device,
            devices=1,
            val_check_interval = args.val_int,        
            logger=logger,
            gradient_clip_val=args.clip, 
        )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)
    model = Net.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes = num_classes, batch_size = batch_size, text_model = datasets_conf[args.ds]["text_model"], text_model_path = datasets_conf[args.ds]["text_model_path"], vision_model = model_conf[args.model]["model"], vision_model_path = model_conf[args.model]["path"][args.ds], lora_r = args.lora_r, lora_alpha = args.lora_alpha, lora_text = args.lora_text, lora_vision = args.lora_vision, weight_decay = args.weight_decay, scheduler =  args.scheduler, lora_dropout =  args.lora_dropout, targets = targets)
    print(trainer.checkpoint_callback.best_model_path)
    #val_acc = trainer.validate(model=model, datamodule=data)[0]['val_acc']
    print(trainer.test(model=model, datamodule=data))
    acc = save_predictions(model, os.path.join(output_dir, 'predictions.csv'), num_classes)
    print("Test accuracy: " + str(acc.item()))
    if args.wandb:
        #wandb_logger.log_metrics({"val_acc": val_acc})
        wandb_logger.log_metrics({"test_acc": acc})