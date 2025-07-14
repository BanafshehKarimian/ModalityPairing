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
from Models.CONCHNet import CONCHNetModel
from Models.PLIPNet import PLIPNetModel
from Models.QUILTNet import QUILTNetModel
from Models.PathCLIPNet import PathCLIPNetModel
from lightning.pytorch.loggers import WandbLogger
import wandb
import torch.nn as nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="CONCH")
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
    parser.add_argument("--lam", type=float, default=1)
    parser.add_argument("--scheduler", type=int, default=0)
    parser.add_argument("--no-lin", action="store_true")
    parser.add_argument("--chkpnt", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--zero-shot", action="store_true")
    args = parser.parse_args()


    output_base_dir = args.dir
    output_name = args.output
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    seed_function(args.sd, extra = False)
    datasets_conf = {"pcam":{"num_classes": 2, "ds_class": PCAMTDataModule, "text_model": TextNetModel, "text_model_path": "./best_models/pcam_TCGA.ckpt"},\
                    "bach":{"num_classes": 4, "ds_class": BACHTDataModule, "text_model": TextNetModel, "text_model_path": "./best_models/bach_TCGA.ckpt"},\
                    "crc":{"num_classes": 9, "ds_class": CRCTDataModule, "text_model": TextNetModel, "text_model_path": "./best_models/crc_TCGA.ckpt"}}
    model_conf = {"CONCH": CONCHNetModel, "QUILT": QUILTNetModel, "PathCLIP": PathCLIPNetModel, "PLIP": PLIPNetModel}
    epochs = args.ep
    device = "gpu" if torch.cuda.is_available() else "cpu"
    
    num_classes = datasets_conf[args.ds]["num_classes"]
    batch_size, num_workers = args.batch, args.worker
    Net = model_conf[args.model]
    '''Loading Data'''
    if args.model in ["UNI", "QUILT", "PLIP", "PathCLIP"] and args.ds in ["bach", "crc", "mhist"]:
        data = datasets_conf[args.ds]["ds_class"](batch_size, num_workers, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    elif args.model in ["CONCH"] and args.ds in ["bach", "crc", "mhist"]:
        data = datasets_conf[args.ds]["ds_class"](batch_size, num_workers, size = 448, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    else:
        data = datasets_conf[args.ds]["ds_class"](batch_size, num_workers, transformer = args.model)
    '''Creating the model'''
    class_names = ["normal", "tumor"]
    if args.ds == "bach":
        class_names = ["normal", "benign", "in situ carcinoma", "invasive carcinoma"]
    elif args.ds == "crc":
        class_names = [
                "Adipose tissue",
                "Background",
                "Debris (necrosis, mucus, hemorrhage)",
                "Lymphocytes",
                "Mucus",
                "Smooth muscle",
                "Normal colon mucosa",
                "Cancer-associated stroma",
                "Colorectal adenocarcinoma epithelium"
            ]
    targets = (nn.Linear, nn.Embedding, nn.Conv2d)
    if args.no_lin:
        targets = (nn.Embedding, nn.Conv2d)
    
    model = Net(class_names, num_classes, batch_size, targets = targets, lora_r = args.lora_r, lora_alpha = args.lora_alpha)

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
    if not args.zero_shot:
        trainer.fit(model, data)
        print(trainer.checkpoint_callback.best_model_path)
        model = Net.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, class_names = class_names, num_classes = num_classes, batch_size = batch_size, targets = targets, lora_r = args.lora_r, lora_alpha = args.lora_alpha)
    val = trainer.validate(model=model, datamodule=data)[0]
    print(trainer.test(model=model, datamodule=data))
    print(val['val_acc'])
    acc = save_predictions(model, os.path.join(output_dir, 'predictions.csv'), num_classes)
    print("Test accuracy: " + str(acc.item()))
    #if args.wandb:
    #    wandb_logger.log_metrics({"val_acc": val['val_acc']})
    #    wandb_logger.log_metrics({"test_acc": acc})
    #with open(output_name + "_lambda.txt", "a") as myfile:
    #    myfile.write(str(args.lora_r) + ", " + str(args.lora_alpha) + ", " + str(args.lam) + ", " + str(val['val_acc']) + ", " + str(val['val_loss']) + ", " + str(acc.item()) + "\n")