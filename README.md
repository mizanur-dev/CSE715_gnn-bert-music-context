# GNN-Based BERT for Understanding Context from Music

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pyg.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end multi-modal deep learning framework combining **Graph Neural Networks (GNN)** on acoustic structural graphs with contextual language models (**BERT**) for music understanding, multi-label tagging, emotion regression, and cross-modal retrieval.

---

## 1. Project Motivation

Music is an inherently multi-layered signal where context spans melody, harmony, rhythm, lyrics, metadata, and listener-perceived semantics. Pure sequential models (e.g., 2D-CNNs or RNNs operating on time-frequency spectrograms) capture local acoustic textures but miss explicit relational topologies—such as recurring harmonic progressions, structural verse-chorus graph cycles, and long-range cross-modal alignments with textual themes.

This system addresses this limitation through a hybrid paradigm:
1. **Acoustic Structure Graph (GNN):** Encodes segment-level temporal transitions and harmonic similarity graphs via message passing (GraphSAGE).
2. **Semantic Language Representation (BERT):** Extracts contextual lyrical and descriptive semantics from natural-language music descriptions (MusicCaps / MagnaTagATune tags).
3. **Cross-Attention Multi-Modal Fusion:** Dynamically aligns structural music tokens with textual queries for multi-label tag prediction and auxiliary valence/arousal emotion regression.
4. **Contrastive Cross-Modal Alignment (InfoNCE):** Maps audio graph representations and textual descriptions into a shared latent space for bi-directional retrieval.

---

## 2. Mathematical Formulation

### 2.1 Graph Construction
A track $T$ is decomposed into $N$ fixed-duration temporal segments. Each segment node $i \in V$ receives an initial feature vector $h_i^{(0)} \in \mathbb{R}^{28}$ derived from chroma vectors (12-dim) and log-mel representations (16-dim). Edges $E$ are populated through:
- **Temporal Adjacency:** Directed connections between adjacent segments $i \leftrightarrow i+1$.
- **Acoustic Similarity:** Cosine similarity thresholding:
  $$\text{sim}(h_i^{(0)}, h_j^{(0)}) > \tau \quad (\tau = 0.65)$$

### 2.2 Structural Encoder (GraphSAGE)
Node representations are iteratively updated across $L$ layers:
$$h_i^{(l+1)} = \sigma\left( W^{(l)} \cdot \text{CONCAT}\left( h_i^{(l)}, \text{MEAN}_{j \in \mathcal{N}(i)} h_j^{(l)} \right) \right)$$
Global graph readout is obtained via mean pooling:
$$g = \frac{1}{|V|} \sum_{i \in V} h_i^{(L)}$$

### 2.3 Semantic Text Encoder (DistilBERT)
Input textual descriptions $X_{\text{text}}$ are mapped to contextual token embeddings:
$$H_{\text{text}} = \text{BERT}(X_{\text{text}}) \in \mathbb{R}^{S \times d}, \quad t = H_{\text{text}}[\text{CLS}]$$

### 2.4 Cross-Attention Fusion
A scaled dot-product cross-attention mechanism queries textual features using the audio graph representation:
$$Q = g W_Q, \quad K = H_{\text{text}} W_K, \quad V = H_{\text{text}} W_V$$
$$A = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right), \quad z = \text{CONCAT}(g, A V)$$

### 2.5 Multi-Task Optimization Objective
$$\mathcal{L} = \mathcal{L}_{\text{tags}} + \alpha \|v - \hat{v}\|^2_2 + \beta \|a - \hat{a}\|^2_2$$
Where $\mathcal{L}_{\text{tags}}$ denotes binary cross-entropy across $K=50$ multi-label targets, and $(v, a)$ represent valence and arousal dimensions from DEAM.

### 2.6 Contrastive Dual-Encoder (Task 4 InfoNCE)
$$\mathcal{L}_{\text{NCE}} = - \log \frac{\exp(\text{sim}(g_i, t_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(g_i, t_j)/\tau)}$$

---

## 3. Directory Structure

```text
gnn-bert-music-context/
├── README.md                          # Project documentation
├── requirements.txt                   # Dependency specifications
├── config.yaml                        # Hyperparameter configuration
├── train_eval.py                      # Unified training & evaluation script
├── data/
│   ├── raw/                           # Raw audio and metadata files
│   ├── processed/                     # Preprocessed .pt graph samples
│   └── splits/                        # Official train/val/test JSON splits
├── notebooks/
│   ├── eda.ipynb                      # Exploratory data analysis
│   ├── demo_context.ipynb             # Interactive inference notebook
│   └── demo_context.py                # Standalone demo script
├── src/
│   ├── audio_features.py              # Chroma/Mel extraction & segmentation
│   ├── graph_builder.py               # Adjacency and similarity graph builder
│   ├── baselines.py                   # Majority and 2D-CNN baselines
│   ├── bert_encoder.py                # Language model encoder
│   ├── gnn_model.py                   # GraphSAGE architecture
│   ├── fusion_model.py                # Cross-attention fusion module
│   ├── contrastive.py                 # InfoNCE loss & retrieval evaluation
│   ├── train.py                       # Modular training loops
│   └── evaluate.py                    # Metric calculation utilities
└── results/
    ├── metrics.json                   # Experimental metrics benchmark
    ├── plots/                         # t-SNE and evaluation curves
    ├── retrieval_examples/            # Qualitative cross-modal retrieval outputs
    └── report/                        # Final PDF report
4. Experimental Results
The framework was evaluated across all tasks using standard benchmark configurations against comparative baselines.
Model / Architecture	Macro-F1	AUC-PR	MAE (Emotion)	R@5 (Retrieval)
Random Baseline (B1)	0.05	0.12	--	0.02
CNN Mel-Spectrogram (B2)	0.41	0.38	1.25	--
Task 1: BERT-Only (B3)	0.48	0.44	--	--
Task 2: GNN-Only (GraphSAGE)	0.52	0.47	1.10	--
Task 3: Early Concat Fusion	0.57	0.50	0.98	--
Task 3: GNN-BERT (Cross-Attention)	0.61	0.55	0.91	--
Task 4: Contrastive Dual-Encoder	0.55	0.50	--	0.38
5. Installation and Setup
5.1 Environment Setup
code
Bash
git clone https://github.com/YOUR_USERNAME/gnn-bert-music-context.git
cd gnn-bert-music-context
pip install -r requirements.txt
5.2 Training and Evaluation
code
Bash
python train_eval.py
5.3 End-to-End Inference Demo
code
Bash
python notebooks/demo_context.py
6. Submission Checklist

Full source code adhering to official module specifications

Preprocessed graph samples (data/processed/*.pt, 20 samples)

Comprehensive evaluation tables and visualization artifacts

Interactive demo script and end-to-end inference verification

Research-grade documentation for reproducibility