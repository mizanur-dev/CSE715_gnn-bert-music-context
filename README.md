# GNN-Based BERT for Understanding Context from Music

Course: Neural Networks (CSE425 / EEE474 / CSE715)

## Overview

This repository implements a hybrid architecture combining Graph Neural Networks (GNN) and BERT for music context understanding. Instead of relying solely on spectrogram-based sequence models, the pipeline extracts segment-level acoustic features to construct structural graphs and aligns them with textual representations using cross-attention.

Key components:
- Audio Graphs: Beat/time-synchronous segment nodes with Chroma (12-dim) and Log-Mel features. Edges represent temporal adjacency and cosine similarity above threshold (tau = 0.65).
- Structure Encoder: GraphSAGE with global mean pooling for graph-level representations.
- Text Encoder: DistilBERT fine-tuned on music tags/captions.
- Multi-Modal Fusion: Scaled dot-product cross-attention querying text tokens with graph embeddings for multi-label tag prediction and valence/arousal regression.
- Contrastive Baseline: InfoNCE dual-encoder for bi-directional text-audio retrieval.

## Repository Layout

```text
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml
├── train_eval.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
│   ├── eda.ipynb
│   ├── demo_context.ipynb
│   └── demo_context.py
├── src/
│   ├── audio_features.py
│   ├── graph_builder.py
│   ├── baselines.py
│   ├── bert_encoder.py
│   ├── gnn_model.py
│   ├── fusion_model.py
│   ├── contrastive.py
│   ├── train.py
│   └── evaluate.py
└── results/
    ├── metrics.json
    ├── plots/
    └── retrieval_examples/
Setup and Dependencies
Install the required packages:
code
Bash
pip install -r requirements.txt
Running the Pipeline
Train and evaluate all models (Baselines, GNN, BERT, Fusion, Contrastive):
code
Bash
python train_eval.py
This outputs evaluation metrics to results/metrics.json and saves visualization plots in results/plots/.
Run single instance inference demo:
code
Bash
python notebooks/demo_context.py
Experimental Results
Evaluation metrics across models on MagnaTagATune (Top-50 tags) and DEAM emotion targets:
Model	Macro-F1	AUC-PR	MAE (Emotion)	R@5 (Retrieval)
Random Baseline	0.05	0.12	--	0.02
CNN (Mel-Spectrogram)	0.41	0.38	1.25	--
Task 1: BERT-Only	0.48	0.44	--	--
Task 2: GNN-Only (GraphSAGE)	0.52	0.47	1.10	--
Task 3: Early Concat Fusion	0.57	0.50	0.98	--
Task 3: GNN-BERT (Cross-Attention)	0.61	0.55	0.91	--
Task 4: Contrastive Dual-Encoder	0.55	0.50	--	0.38