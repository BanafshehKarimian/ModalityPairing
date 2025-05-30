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
python train_clipit.py --dataset PCAM --backbone vit_b16 --lora_rank 16 --lora_alpha 4
```

You can change parameters such as:

- `--backbone`: Vision backbone (ViT-B/16, DINO, UNI, etc.)
- `--text_encoder`: Pretrained CLIP text encoder (e.g., Conch)
- `--lora_*`: LoRA parameters for efficient fine-tuning

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
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2025}
}
```

---

## 🤝 Acknowledgements

This work is supported by the Canadian Institutes of Health Research, NSERC, and the Digital Research Alliance of Canada.
