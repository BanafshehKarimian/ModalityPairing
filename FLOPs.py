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
from Models.LateFuser_ import FuserModel_test as FuserModel
from torchmetrics.functional import auroc
import numpy as np
import random
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
import argparse
from utils.utils import *
from Models.TextNetC import TextNetModel
#from Models.TextNetP import TextNetModel
from Models.UNINet import UNINetModel
from Models.DINOL14Net import DINOL14NetModel
from Models.VITSNet import VITSNetModel8, VITSNetModel16
from Models.VITBNet import VITBNetModel8, VITBNetModel16
from Models.CONCHVisionNet import CONCHVisionNetModel
from Models.PLIPVisionNet import PLIPVisionNetModel
from Models.QUILTVisionNet import QUILTVisionNetModel
from lightning.pytorch.loggers import WandbLogger
import wandb
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from Models.CONCHNet import CONCHNetModel
from Models.PLIPNet import PLIPNetModel
from Models.QUILTNet import QUILTNetModel
from calflops import calculate_flops

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="UNI")
    parser.add_argument("--type", type=str, default="CLIPIT")
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
    parser.add_argument("--save-embd", action="store_true")
    args = parser.parse_args()


    output_base_dir = args.dir
    output_name = args.output
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.wandb:
        wandb.init(project=args.ds + "_" + args.model, reinit=True)
        wandb.config.update(args)

    
    seed_function(args.sd)#output/pcam_text/version_8/checkpoints/epoch=4-step=19247.ckpt
    datasets_conf = {"pcam":{"num_classes": 2, "ds_class": PCAMTDataModule, "text_model": TextNetModel, "text_model_path": "./best_models/pcam_TCGA.ckpt"},\
                    "bach":{"num_classes": 4, "ds_class": BACHTDataModule, "text_model": TextNetModel, "text_model_path": "./best_models/bach_TCGA.ckpt"},\
                    "crc":{"num_classes": 9, "ds_class": CRCTDataModule, "text_model": TextNetModel, "text_model_path": "./best_models/crc_TCGA.ckpt"}}
    model_conf = {"UNI": {"model": UNINetModel, "path": {"pcam": "./best_models/run1/pcam_UNI.ckpt", "bach": "./best_models/run1/bach_UNI.ckpt", "crc": "./best_models/run1/crc_UNI.ckpt"}},\
                    "CONCH": {"model": CONCHVisionNetModel, "path": {"pcam": None, "bach": None, "crc": None}},\
                    "PLIP": {"model": PLIPVisionNetModel, "path": {"pcam": None, "bach": None, "crc": None}},\
                    "QUILT": {"model": QUILTVisionNetModel, "path": {"pcam": None, "bach": None, "crc": None}},\
                    "DINOL14": {"model": DINOL14NetModel, "path": {"pcam": "./best_models/run1/pcam_DINOL14.ckpt", "bach": "./best_models/run1/bach_vitl14.ckpt", "crc": "./best_models/run1/crc_vitl14.ckpt"}},\
                    "VITS_8": {"model": VITSNetModel8, "path": {"pcam": "./best_models/run1/pcam_VITS_8.ckpt", "bach": "./best_models/run1/bach_vits8.ckpt", "crc": "./best_models/run1/crc_vits8.ckpt"}},\
                    "VITS_16": {"model": VITSNetModel16, "path": {"pcam": "./best_models/run1/pcam_VITS_16.ckpt", "bach": "./best_models/run1/bach_vits16.ckpt", "crc": "./best_models/run1/crc_vits16.ckpt"}},\
                    "VITB_8": {"model": VITBNetModel8, "path": {"pcam": "./best_models/run1/pcam_VITB_8.ckpt", "bach": "./best_models/run1/bach_vitb8.ckpt", "crc": "./best_models/run1/crc_vitb8.ckpt"}},\
                    "VITB_16": {"model": VITBNetModel16, "path": {"pcam": "./best_models/run1/pcam_VITB_16.ckpt", "bach": "./best_models/run1/bach_vitb16.ckpt", "crc": "./best_models/run1/crc_vitb16.ckpt"}}}
    
    vision_models = {"UNI": UNINetModel,\
                "QUILT": QUILTVisionNetModel,\
                "CONCH": CONCHVisionNetModel,\
                "PLIP": PLIPVisionNetModel,\
                    "DINOL14": DINOL14NetModel,\
                    "VITS_8": VITSNetModel8, \
                    "VITS_16": VITSNetModel16,\
                    "VITB_8":  VITBNetModel8, \
                    "VITB_16": VITBNetModel16}
    learners = {"late_fusion": FuserModel, "late_fusion_kd": FuserModel}

    epochs = args.ep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = datasets_conf[args.ds]["num_classes"]
    batch_size, num_workers = args.batch, args.worker
    '''Loading Data'''
    if args.model in ["UNI", "QUILT"] and args.ds in ["bach", "crc", "mhist"]:
        data = datasets_conf[args.ds]["ds_class"](batch_size, num_workers, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    elif args.model in ["CONCH"] and args.ds in ["bach", "crc", "mhist"]:
        data = datasets_conf[args.ds]["ds_class"](batch_size, num_workers, size = 448, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    else:
        data = datasets_conf[args.ds]["ds_class"](batch_size, num_workers, transformer = args.model)
    '''Creating the model'''
    targets = (nn.Linear, nn.Embedding, nn.Conv2d)
    if args.no_lin:
        targets = (nn.Embedding, nn.Conv2d)
    if model_conf[args.model]["path"][args.ds]:
        model_conf[args.model]["path"][args.ds] = model_conf[args.model]["path"][args.ds].replace("run1", args.run)
    if args.type in ["CLIPIT"]:
        Net = learners[args.learner]
        model = Net(num_classes, batch_size, targets = targets, lr = args.lr, text_model = datasets_conf[args.ds]["text_model"], text_model_path = datasets_conf[args.ds]["text_model_path"], vision_model = model_conf[args.model]["model"], vision_model_path = model_conf[args.model]["path"][args.ds], lora_r = args.lora_r, lora_alpha = args.lora_alpha, lora_text = args.lora_text, lora_vision = args.lora_vision, weight_decay = args.weight_decay, scheduler = args.scheduler, lora_dropout =  args.lora_dropout, lam = args.lam)
    elif args.type in ["vision"]:
        Net = vision_models[args.model]
        model = Net(num_classes, batch_size, lora = True, targets = targets, lora_r = args.lora_r, lora_alpha = args.lora_alpha)
    else:
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
        model_conf = {"CONCH": CONCHNetModel, "QUILT": QUILTNetModel}
        Net = model_conf[args.model]
        model = Net(class_names, num_classes, batch_size, targets = targets, lora_r = args.lora_r, lora_alpha = args.lora_alpha)
    
    
    dummy_input = torch.randn(1, 3, 224, 224)
    model.eval()
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    '''
    flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(1, 3, 224, 224),
                                      output_as_string=True,
                                      output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    '''
    flops = FlopCountAnalysis(model, dummy_input)
    print(f"Total FLOPs: {flops.total():,.0f}")
    print(f"FLOPs by operation:\n{flops.by_operator()}")

    # Count Parameters
    print(parameter_count_table(model))