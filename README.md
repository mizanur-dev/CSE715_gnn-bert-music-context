# GNN-Based BERT for Understanding Context from Music

Supervised Neural Network Project combining BERT and Graph Neural Networks (GNN) for music context understanding.

## Project Structure
- src/: Core Python modules
- data/processed/: 20 preprocessed .pt music structure graphs
- data/splits/: Official train/val/test JSON splits
- notebooks/: EDA and End-to-end Inference demo notebooks
- results/: Performance metrics JSON, evaluation plots, and retrieval examples

## Setup and Execution
```bash
pip install -r requirements.txt
python train_eval.py
python notebooks/demo_context.py
```