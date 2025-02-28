import os
import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import auroc
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import f1_score
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
import random
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import glob, os
import pickle
import pandas as pd
import shutil
import re
from collections import Counter
import matplotlib.pyplot as plt 
import seaborn as sns
from torchmetrics import classification

def clean_pathology_report(report: str) -> str:
    """
    Cleans an OCR-generated pathology report by removing redundant whitespace,
    non-informative characters, and unnecessary repetitions.

    Args:
        report (str): The raw OCR-generated pathology report text.

    Returns:
        str: The cleaned and formatted pathology report.
    """
    if report is None:
        return report
    # Remove redundant line breaks and excessive spaces
    cleaned_report = re.sub(r'\n+', '\n', report)  # Replace multiple newlines with one
    # Remove sequences of mixed "I" and "1" characters
    cleaned_report = re.sub(r'[1|I|i]*', '', cleaned_report)
    cleaned_report = re.sub(r'[ \t]+', ' ', cleaned_report)  # Replace multiple spaces with one
    # Remove non-informative repeated characters (e.g., "IIIIIIII")
    cleaned_report = re.sub(r'[I|l|\|]{5,}', '', cleaned_report)
    # Remove sequences of mixed "I" and "1" characters
    cleaned_report = re.sub(r'[1I]{5,}', '', cleaned_report)
    # Remove artifacts like placeholders or non-alphanumeric noise
    cleaned_report = re.sub(r'[\x00-\x1F\x7F]+', '', cleaned_report)  # Remove control characters
    cleaned_report = re.sub(r'[\-]{2,}', '-', cleaned_report)  # Normalize long dashes
    cleaned_report = re.sub(r'[\*]+', '', cleaned_report)  # Remove asterisks
    # Remove headings/fields with excessive underscores or placeholders
    cleaned_report = re.sub(r'_+', '', cleaned_report)
    cleaned_report = re.sub(r'Redacted', '', cleaned_report)
    # Standardize certain phrases for readability
    cleaned_report = re.sub(r'Mets', 'Metastases', cleaned_report, flags=re.IGNORECASE)
    cleaned_report = re.sub(r'Histologic Type:', '\nHistologic Type:', cleaned_report)
    # Ensure proper spacing after punctuation
    cleaned_report = re.sub(r'([.,;:])([^ \n])', r'\1 \2', cleaned_report)
    # Remove extra spaces around newlines
    cleaned_report = re.sub(r'\s*\n\s*', '\n', cleaned_report)
    # Trim leading and trailing whitespace
    cleaned_report = cleaned_report.strip()
    return cleaned_report


def seed_function(seed, extra = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)

    # If you're using GPUs:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if extra:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(1)

def save_predictions(model, output_fname, num_classes):
    prds = torch.cat(model.predictions, dim=0)
    trgs = torch.cat(model.targets, dim=0)
    acc = classification.MulticlassAccuracy(
                    num_classes=num_classes,
                    average="macro",
                )
    acc = acc(prds.cpu(), trgs.cpu().type(torch.int64))

    auc = auroc(prds, trgs, num_classes=num_classes, average='macro', task='multiclass')

    print('AUROC (test)')
    print(auc)

    cols_names = ['class_' + str(i) for i in range(0, num_classes)]

    df = pd.DataFrame(data=prds.cpu().numpy(), columns=cols_names)    
    df['target'] = trgs.cpu().numpy()
    df.to_csv(output_fname, index=False)
    l = []
    for i in range(num_classes):
        l.append(df['class_'+ str(i)])
    preds = np.stack(l).transpose()
    targets = np.array(df['target'])
    print("balanced accuracy, F1 score:")
    print(acc, accuracy_score(targets, preds.argmax(1)), f1_score(targets, preds.argmax(1), average='micro'))
    return acc

from sklearn.manifold import TSNE
def density_plot(vision_embeddings_before, vision_embeddings_after, text_embeddings_before, text_embeddings_after, save_path):
    tsne_v = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_v_ = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_t = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_t_ = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    vision_after_tsne = tsne_v.fit_transform(vision_embeddings_after)
    vision_before_tsne = tsne_v_.fit_transform(vision_embeddings_before)
    text_after_tsne = tsne_t.fit_transform(text_embeddings_after)
    text_before_tsne = tsne_t_.fit_transform(text_embeddings_before)
    ud_a = np.linalg.norm(vision_before_tsne - text_before_tsne, axis = 1)
    ud_b = np.linalg.norm(vision_after_tsne - text_after_tsne, axis = 1)
    plt.figure(figsize=(10, 6))
    plt.hist(ud_a, bins=50, alpha=0.7, label='Before Training', density=False)
    plt.hist(ud_b, bins=50, alpha=0.7, label='After Training', density=False)
    plt.xlabel('L2 distance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    # Save the figure
    plt.savefig(save_path)
    plt.close()


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


import torch.nn as nn

def replace_linear_with_lora(model, rank, alpha):
    """
    Replace all nn.Linear layers in the model with LinearWithLoRA layers.
    
    Args:
        model (nn.Module): The PyTorch model.
        rank (int): Rank parameter for the LoRA layer.
        alpha (float): Alpha parameter for the LoRA layer.

    Returns:
        nn.Module: The model with LinearWithLoRA layers replacing Linear layers.
    """
    for name, module in model.named_children():
        # Recursively apply to child modules
        if isinstance(module, nn.Linear):
            # Replace Linear with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)
    
    return model


def plot_filtered_confusion_matrix(num_classes=2, save_path = 'bar.png'):
    # Load Text Model Predictions
    t = pd.read_csv('csv/pcam_text.csv')
    preds_t = np.stack([t[f'class_{i}'] for i in range(num_classes)]).transpose().argmax(axis=1)
    targets = np.array(t['target'])
    t_corr = preds_t == targets
    bars = []
    vision_models = ['UNI', 'DINO', 'VITS_16', 'VITS_8', 'VITB_16', 'VITB_8']

    # Load Vision Model Predictions
    for model in vision_models:
        v = pd.read_csv('csv/pcam_'+model+'.csv')
        preds_v = np.stack([v[f'class_{i}'] for i in range(num_classes)]).transpose().argmax(axis=1)
        v_corr = preds_v == targets
        bars.append((t_corr*(1-v_corr)).sum())
    plt.figure(figsize=(10, 6))
    plt.bar(vision_models, bars, color='skyblue', alpha=0.8)
    plt.xlabel("Vision Models")
    plt.ylabel("Number of Incorrect Predictions")
    plt.title("Vision Model Errors on Text Model Correct Predictions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
