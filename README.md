# CLIP-IT: CLIP-based Pairing of Histology Images with Privileged Textual Information

[![Paper](https://img.shields.io/badge/MICCAI-2025-blue)](https://doi.org/...)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![Code](https://img.shields.io/badge/code-available-brightgreen)](https://github.com/BanafshehKarimian/ModalityPairing)

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

Ensure you have the required datasets (PCAM, BACH, CRC) and optionally TCGA reports for pairing.

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

| Backbone     | PCAM         | BACH         | CRC          |
|--------------|--------------|--------------|--------------|
| UNI          | 94.24 → 95.49 | 78.89 → 81.79 | 94.66 → 95.92 |
| DINO         | 88.88 → 92.32 | 84.26 → 86.11 | 94.40 → 95.91 |
| ViT-B/16     | 88.13 → 91.42 | 80.78 → 82.94 | 95.86 → 95.67 |

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

## 🤝 Acknowledgements

This work is supported by the Canadian Institutes of Health Research, NSERC, and the Digital Research Alliance of Canada.
