from Data.PCAMTDataLoaderConch import PCAMTDataModule 
from Data.PCAMTDataLoaderConchPrompt import PCAMTDataModule as PCAMTDataModule_
from Data.PCAMTDataLoaderConchPrompttqx import PCAMTDataModule as PCAMTDataModule__
from Data.BACHTDataLoaderConch import BACHTDataModule
from Data.BACHTDataLoaderConchPrompt import BACHTDataModule as BACHTDataModule_
from Data.BACHTDataLoaderConchPrompttqx import BACHTDataModule as BACHTDataModule__
from Data.CRCTDataLoaderConch import CRCTDataModule
from Data.CRCTDataLoaderConchPrompt import CRCTDataModule as CRCTDataModule_
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from Models.TextNetC import TextNetModel
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
import wandb
from utils.utils import *




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="pcam")
    parser.add_argument("--dir", type=str, default='output')
    parser.add_argument("--output", type=str, default='train_text')
    parser.add_argument("--sd", type=int, default=0)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--bach", type=int, default=64)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--chkpnt", type=str, default='')
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--tqx", action="store_true")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--text-k", type=int, default=1)
    args = parser.parse_args()

    device = "gpu" if torch.cuda.is_available() else "cpu"
    pl.seed_everything(0, workers=True)
    batch_size, num_workers = args.bach, args.worker
    '''Loading Data'''
    data_name = args.ds
    if data_name == "pcam":
        num_classes = 2
        DataModule = PCAMTDataModule
        if args.prompt:
            DataModule = PCAMTDataModule_
        data = DataModule(batch_size, num_workers)
        if args.tqx:
            DataModule = PCAMTDataModule__
            data = DataModule(batch_size, num_workers, level = args.level)
        epochs = 5
    elif data_name == "bach":
        num_classes = 4
        DataModule = BACHTDataModule
        if args.prompt:
            DataModule = BACHTDataModule_
        data = DataModule(batch_size, num_workers)
        if args.tqx:
            DataModule = BACHTDataModule__
            data = DataModule(batch_size, num_workers, level = args.level)
        epochs = 100
    else:
        num_classes = 9
        DataModule = CRCTDataModule
        if args.prompt:
            DataModule = CRCTDataModule_
        epochs = 10
        data = DataModule(batch_size, num_workers)

    output_base_dir = 'output'
    output_name = data_name + '_text'
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    '''Creating the model'''
    Net = TextNetModel
    model = Net(num_classes, batch_size, k = args.text_k)

    print('=============================================================')
    print('Training...')

    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max')
    early_stop_callback = EarlyStopping(
            monitor='val_acc',
            min_delta=0.001,
            patience=10,  # NOTE no. val epochs, not train epochs
            verbose=False,
            mode="max",
        )
    logger = [TensorBoardLogger(output_base_dir, name=output_name)]
    if args.wandb:
        wandb.init(project="text_finetuning", reinit=True)
        wandb.config.update(args)
        wandb_logger = WandbLogger(log_model=False)
        logger = [wandb_logger, TensorBoardLogger(output_base_dir, name=output_name)]
    trainer = pl.Trainer(
            callbacks=[checkpoint_callback],#, early_stop_callback
            log_every_n_steps=1,
            max_epochs=epochs,
            accelerator=device,
            devices=1,
            val_check_interval = 0.1,  
            logger=logger,
        )
    trainer.logger._default_hp_metric = False
    if not args.test:
        trainer.fit(model, data)
        model = Net.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes = num_classes, k = args.text_k)
        print(trainer.checkpoint_callback.best_model_path)
    else:
        checkpoint = torch.load(args.chkpnt)
        model.load_state_dict(checkpoint['state_dict'])
    print(trainer.test(model=model, datamodule=data))
    save_predictions(model, os.path.join(output_dir, 'predictions.csv'), num_classes)