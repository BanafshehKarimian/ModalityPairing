# CLIP-IT: CLIP-based Pairing of Histology Images with Privileged Textual Information

> **CLIP-IT** is a novel framework for enhancing histology image classification by **pairing vision data with external, unpaired text reports** using CLIP-based matching. It **trains a unimodal classifier with multimodal benefits** — without requiring text at inference time.

---

## 🔍 Overview

Current multimodal vision-language models (VLMs) for cancer diagnosis rely on expensive, manually paired datasets of histology images and pathology reports. CLIP-IT tackles this bottleneck by:

- **Pairing** histology images with relevant external reports using a CLIP model.
- **Distilling** knowledge from the text modality into the vision model using feature-level distillation.
- **Discarding** the text modality at inference time — fast and efficient deployment.

<p align="center">
  <img src="clipit-diagram.png" alt="CLIP-IT Diagram" width="600"/>
</p>

---

## 🧠 Core Contributions

- ✅ Use of a **CLIP-based retrieval system** to match histology images with semantically related external reports  
- ✅ Training via **multimodal fusion and distillation** to enhance vision-only classifiers  
- ✅ Final model is **unimodal at test time** — no access to text required  
- ✅ Compatible with **any vision backbone and any unpaired textual corpus**

---

## 📦 Installation

```bash
git clone https://github.com/BanafshehKarimian/ModalityPairing.git
cd ModalityPairing
pip install -r requirements.txt
```

Ensure you have the required datasets (PCAM, BACH, CRC) and optionally TCGA reports for pairing. Put them here /export/datasets/public/.

---

## 🧪 Datasets Used

| Dataset | Description                | Classes | Patch Size   | Magnification |
|---------|----------------------------|---------|--------------|----------------|
| PCAM    | Breast tissue              | 2       | 96×96        | 10x            |
| BACH    | Breast cancer histology    | 4       | 2048×1536    | 20x            |
| CRC     | Colorectal cancer          | 9       | 224×224      | 20x            |

External text modality: TCGA pathology reports.

---

## 🚀 Training
First download the conch checkpoints and put it in the following location:

```bash
CONCH/checkpoints/conch/pytorch_model.bin
```

Then get the text:
```bash
python get_text.py --keyword tissue_type
```
The paired indexes are in the Paired_indexes folder, but for any dataset, you can use a method as shown in map_text_pcam.py. 

Then you need to fine-tune the text model:
```bash
python finetune_text.py --ds pcam 
```

And finally train your CLIPIT:

```bash
python train_fuser_.py --ds pcam --model UNI --lora_r 16 --lora_alpha 4
```

## ⚙️ Script Arguments

The main training script supports a wide range of arguments for flexible configuration:

### 📁 Dataset and Experiment Settings

- `--model`: Vision backbone (UNI, DINOL14, VITS_8, VITS_16, VITB_8, VITB_16)
- `---ds`: Dataset (pcam, bach, crc)
- `--run`: Run identifier for experiment versioning
- `--dir`: Directory to save logs and checkpoints
- `--output`: Output folder for this run

### 🔧 Training Hyperparameters
- `--batch`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--ep`: Number of training epochs
- `--worker`: Number of data loader workers
- `--val-int`: Fraction of training data for validation (e.g., 0.1 for 10%)
- `--clip`: Gradient clipping value (default: 0.5)
- `--weight-decay`: Weight decay for optimizer
- `--scheduler`: Whether to use learning rate scheduler (0 or 1)

### 🧠 LoRA Configuration
- `--lora-r`: LoRA rank (e.g., 16)
- `--lora-alpha`: LoRA scaling factor
- `--lora-dropout`: Dropout used in LoRA modules
- `--lora-text`: Apply LoRA to text encoder (1 = yes, 0 = no)
- `--lora-vision`: Apply LoRA to vision encoder (1 = yes, 0 = no)
- `--no-lin`: Remove linear layer from LoRA targets
- 
### 🛠 Optimization and Logging
- `--monitor`: Metric to monitor (e.g., `val_loss`)
- `--mod`: Mode for monitoring (`min` or `max`)
- `--patience`: Early stopping patience
- `--early`: Enable early stopping
- `--log`: Logging frequency in epochs
- `--wandb`: Enable Weights & Biases logging
- `--chkpnt`: Save model checkpoints
---

## 📈 Results

### Classification Accuracy (± std) Averaged Over Three Runs  
*Comparing unimodal backbones and their CLIP-IT-enhanced counterparts across PCAM, BACH, and CRC datasets.*

#### Unimodal Backbones

| Backbone           | PCAM (Unimodal) | PCAM (CLIP-IT) | Δ     | BACH (Unimodal) | BACH (CLIP-IT) | Δ     | CRC (Unimodal) | CRC (CLIP-IT) | Δ     |
|--------------------|------------------|----------------|-------|------------------|----------------|-------|----------------|----------------|-------|
| UNI                | 94.24 ± 0.14     | **95.49 ± 0.27** | **+1.3** | 78.89 ± 1.56     | **81.79 ± 1.98** | **+2.9** | 94.66 ± 0.41     | **95.92 ± 0.07** | **+1.3** |
| DINO               | 88.88 ± 0.75     | **92.32 ± 0.84** | **+3.4** | 84.26 ± 2.30     | **86.11 ± 3.21** | **+1.8** | 94.40 ± 0.40     | **95.91 ± 0.06** | **+1.6** |
| VITB-16            | 88.13 ± 1.09     | **91.42 ± 1.28** | **+3.3** | 80.78 ± 2.14     | **82.94 ± 4.52** | **+2.1** | **95.86 ± 0.25** | 95.67 ± 0.18     | -0.2  |
| VITS-16            | 88.43 ± 0.26     | **90.93 ± 1.12** | **+2.5** | 82.90 ± 5.56     | **84.64 ± 6.68** | **+1.7** | 93.77 ± 0.29     | **95.34 ± 0.20** | **+1.7** |
| VITB-8             | 87.54 ± 0.71     | **91.92 ± 0.87** | **+4.4** | 86.90 ± 2.32     | **87.06 ± 1.22** | **+0.2** | **95.71 ± 0.11** | 95.66 ± 0.53     | -0.1  |
| VITS-8             | 87.90 ± 0.61     | **90.24 ± 0.90** | **+2.3** | 81.01 ± 3.04     | **82.55 ± 1.57** | **+1.5** | **95.03 ± 0.13** | 94.80 ± 0.60     | -0.2  |

#### Multimodal Backbones

| Backbone     | PCAM (CLIP-IT) | PCAM (Contrastive) | PCAM (Vision) | BACH (CLIP-IT) | BACH (Contrastive) | BACH (Vision) | CRC (CLIP-IT) | CRC (Contrastive) | CRC (Vision) |
|--------------|----------------|--------------------|----------------|----------------|---------------------|----------------|----------------|---------------------|----------------|
| CONCH        | **93.61 ± 0.44** | 92.67 ± 1.40        | 91.75 ± 2.57    | **85.05 ± 0.64** | 60.78 ± 0.29         | 67.25 ± 4.34    | 94.89 ± 0.61     | **95.58 ± 0.40**      | 95.12 ± 0.47    |
| QUILTNet     | **91.83 ± 2.37** | 89.82 ± 0.62        | 90.44 ± 0.65    | **65.50 ± 1.84** | 55.82 ± 5.86         | 63.81 ± 2.16    | 94.83 ± 0.94     | **95.37 ± 0.17**      | 94.60 ± 0.65    |
CLIP-IT yields consistent performance gains with minimal inference overhead.

---

## 🔬 Citation

If you use this work, please cite:

```bibtex
@inproceedings{karimian2025clipit,
  title={CLIP-IT: CLIP-based Pairing of Histology Images with Privileged Textual Information},
  author={Karimian, Banafsheh and Avanzato, Giulia and Belharbi, Soufian and McCaffrey, Luke and Shateri, Mohammadhadi and Granger, Eric},
  booktitle={arxiv},
  year={2025}
}
```

---
