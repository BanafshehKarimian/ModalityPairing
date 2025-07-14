from Data.PCAMTDataLoaderConch import PCAMTDataModule as PCAMDataModule
from Data.BACHTDataLoaderConch import BACHTDataModule as BACHDataModule
from Data.CRCTDataLoaderConch import CRCTDataModule as  CRCDataModule
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import argparse
from sklearn.utils import shuffle
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import auroc
from Models.UNINet import UNINetModel
from Models.DINOL14Net import DINOL14NetModel
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from Models.VITBNet import VITBNetModel8, VITBNetModel16
from Models.VITSNet import VITSNetModel8, VITSNetModel16
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torchmetrics import classification
from Models.CONCHVisionNet import CONCHVisionNetModel
from Models.PathCLIPVisionNet import PathCLIPVisionNetModel
from Models.QUILTVisionNet import QUILTVisionNetModel
from Models.PLIPVisionNet import PLIPVisionNetModel
from utils.utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="UNI")
    parser.add_argument("--ds", type=str, default="pcam")
    parser.add_argument("--dir", type=str, default='output')
    parser.add_argument("--output", type=str, default='train_vision')
    parser.add_argument("--sd", type=int, default=0)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--chkpnt", type=str, default='')
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-lin", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--add-lora", action="store_true")
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--val-int", type=float, default=0.1)
    args = parser.parse_args()


    output_base_dir = args.dir
    output_name = args.output# + '_' + args.ds + '_' + args.model
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = args.ds
    model = args.model

    seed_function(args.sd)
    num_workers = args.worker
    device = "gpu" if torch.cuda.is_available() else "cpu"
    datasets_conf = {"pcam":{"num_classes": 2, "ds_class": PCAMDataModule},\
                    "bach":{"num_classes": 4, "ds_class": BACHDataModule},\
                    "crc":{"num_classes": 9, "ds_class": CRCDataModule}}
    models = {"UNI": UNINetModel,\
                "QUILT": QUILTVisionNetModel,\
                "CONCH": CONCHVisionNetModel,\
                "PathCLIP": PathCLIPVisionNetModel,\
                "PLIP": PLIPVisionNetModel,\
                    "DINOL14": DINOL14NetModel,\
                    "VITS_8": VITSNetModel8, \
                    "VITS_16": VITSNetModel16,\
                    "VITB_8":  VITBNetModel8, \
                    "VITB_16": VITBNetModel16}
    
    wandb_logger = WandbLogger(project= data + "_" + model + "_vision_only" , name= output_name, offline = True)
    logger = [TensorBoardLogger(output_base_dir, name=output_name)]
    if args.wandb:
        logger.append(wandb_logger)
    Net = models[model]

    if data == "pcam":
        max_steps = 4100
        num_classes = 2
        batch_size = 64
        DataModule = PCAMDataModule
        lr=0.001
        momentum=0.9
        nesterov = True
        weight_decay = 0.0001
        bbfroze = True
        min_delta = 0 
        patience = 9
        monitor="val_acc"
        mode='max'
        #BATCH_SIZE = 4096, PREDICT_BATCH_SIZE = 64, N_RUNS =  5, BCEWithLogitsLoss, 
    elif data == "bach":
        max_steps = 25 * 4
        num_classes = 4
        batch_size = 64
        DataModule = BACHDataModule
        lr=0.001
        momentum=0.9
        nesterov = True
        weight_decay = 0.0001
        bbfroze = True
        min_delta = 0 
        patience = 9
        monitor="val_acc"
        mode='max'
        #BATCH_SIZE = 256, PREDICT_BATCH_SIZE = 64, N_RUNS =  5
    elif data == "crc":
        max_steps = 1251
        num_classes = 9
        batch_size = 64 #256
        DataModule = CRCDataModule
        lr=0.001
        momentum=0.9
        nesterov = True
        weight_decay = 0.0001
        bbfroze = True
        min_delta = 0 
        patience = 9
        monitor="val_acc"
        mode='max'
        #BATCH_SIZE = 4096, PREDICT_BATCH_SIZE = 64, N_RUNS =  5
    '''Loading Data'''
    if args.model in ["UNI", "PathCLIP", "PLIP"] and args.ds in ["bach"]:
        data = DataModule(batch_size, num_workers, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], no_text=True)
    elif args.model in ["PLIP"]:
        data = DataModule(batch_size, num_workers, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], no_text=True)
    elif args.ds in ["bach", "crc"]:
        data = DataModule(batch_size, num_workers, no_text=True)
    else:
        data = DataModule(batch_size, num_workers, no_text=True, transformer = args.model)
    output_base_dir = 'output'
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    '''Creating the model'''
    targets = (nn.Linear, nn.Embedding, nn.Conv2d)
    if args.no_lin:
        targets = (nn.Embedding, nn.Conv2d)
    model = Net(num_classes, batch_size, lr, momentum, nesterov, weight_decay, bbfroze, lora = args.add_lora, targets = targets, lora_r = args.lora_r, lora_alpha = args.lora_alpha)

    print('=============================================================')
    print('Training...')

    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=mode)
    early_stop_callback = EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,  # NOTE no. val epochs, not train epochs
            verbose=False,
            mode=mode,
        )
    trainer = pl.Trainer(
            callbacks = [checkpoint_callback],
            log_every_n_steps=1,
            max_epochs=args.ep,
            accelerator=device,
            devices=1,
            val_check_interval = args.val_int,        
            logger=logger,
            gradient_clip_val=args.clip, 
        )
    trainer.logger._default_hp_metric = False
    if not args.test:
        trainer.fit(model, data)
        print(trainer.checkpoint_callback.best_model_path)# = 'output/DINOL14_bach/version_0/checkpoints/epoch=60-step=122.ckpt'
        model = Net.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes = num_classes, batch_size = batch_size, lora = args.add_lora, targets = targets, lora_r = args.lora_r, lora_alpha = args.lora_alpha)
    else:
        vision_model_path = args.chkpnt
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
        print("done loading")
        model.load_state_dict(new_state_dict)
    val = trainer.validate(model=model, datamodule=data)[0]
    print(trainer.test(model=model, datamodule=data))
    acc = save_predictions(model, os.path.join(output_dir, 'predictions.csv'), num_classes)
    print(val['val_acc'])
    print("Test accuracy: " + str(acc.item()))
    