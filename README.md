# Vision Transformer (ViT) Classifier from Scratch

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-from%20scratch-red?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/dataset-CIFAR--10-orange)
![License](https://img.shields.io/badge/license-MIT-yellow)

A clean PyTorch implementation of the Vision Transformer (ViT) architecture from scratch, trained and evaluated on the CIFAR-10 dataset. This repository contains the complete network blocks (Patch Embedding, Multi-head Self-Attention, Transformer Encoder, and Classification Head) along with training loops, learning rate scheduling, and accuracy plotting utilities.

---

## 🚀 Key Features

* **ViT Blocks from Scratch**: Custom implementation of:
  * **Patch Embedding**: Splits 2D images into patches and projects them to a 1D embedding dimension.
  * **Multi-head Self-Attention (MHSA)**: Implements queries, keys, and values matrices, calculating scaled dot-product attention across heads.
  * **Transformer Encoder**: Combines MHSA with Layer Normalization, Residual Connections, and Multilayer Perceptron (MLP) layers.
* **Flexible Classification Modes**: Support for both Class Token (CLS Token) pooling and Global Average Pooling (GAP) classifier interfaces.
* **Training Pipeline**: Script with learning rate scheduler, weight decay optimizations, validation metrics, and model checkpoint saving.
* **Plotting Utility**: Generates clean visual graphs of training vs. validation loss and accuracy curves over epochs.

---

## 🛠️ Tech Stack
* **Framework**: PyTorch
* **Data & CV**: torchvision, Matplotlib, NumPy, JSON

---

## 📁 Repository Structure
```text
├── models.py       # Custom Vision Transformer layers (PatchEmbedding, Attention, EncoderBlock, ViT)
├── train.py        # Dataset loading (CIFAR-10), training loop, and evaluation
├── plot.py         # Matplotlib helper to visualize loss and accuracy curves
├── run_all.py      # Automates experiment executions across parameter grids
└── README.md
```

---

## ⚙️ Installation & Usage

1. **Install PyTorch and standard dependencies**:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```

2. **Train the Vision Transformer**:
   To train with Class Token pooling:
   ```bash
   python train.py --epochs 20 --batch_size 128 --lr 0.001 --weight_decay 0.05
   ```
   To train using Global Average Pooling (GAP):
   ```bash
   python train.py --epochs 20 --use_gap --lr 0.001
   ```

3. **Plot Training Metrics**:
   To generate loss and accuracy graphs:
   ```bash
   python plot.py --history_path checkpoints/history.json --output_dir plots/
   ```

---

## 📐 Architecture Details

The model processes an image $x \in \mathbb{R}^{H \times W \times C}$ by reshaping it into a sequence of flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $P$ is the patch size and $N = HW/P^2$ is the sequence length. These patches are projected to `embed_dim` via `PatchEmbedding`.
An learnable CLS token is prepended to the sequence, and standard 1D positional embeddings are added before passing the sequence through $L$ consecutive `TransformerEncoder` blocks.
```text
[Image] -> [Patch Embedding] -> [Add CLS Token & Position Embs] -> [Transformer Encoder Blocks x L] -> [MLP Head] -> [Class Logits]
```


---

<sub>📦 Part of my <a href="https://github.com/Nurassyl-labs/ai-ml-portfolio">AI/ML portfolio</a>.</sub>
