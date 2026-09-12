import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from transformers import AutoTokenizer
from src.graph_builder import build_segment_graph
from src.gnn_model import MusicGraphSAGE
from src.bert_encoder import MusicBERTClassifier
from src.fusion_model import GNNBERTFusion

def run_inference(caption_text, num_classes=50):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokens = tokenizer(caption_text, return_tensors="pt", padding=True, truncation=True)

    dummy_feats = np.random.randn(6, 28).astype(np.float32)
    graph = build_segment_graph(dummy_feats)
    batch_idx = torch.zeros(graph.x.size(0), dtype=torch.long)

    bert = MusicBERTClassifier(num_classes=num_classes)
    gnn = MusicGraphSAGE(in_channels=28, hidden_dim=128, num_classes=num_classes)
    fusion = GNNBERTFusion(g_dim=128, t_dim=768, num_classes=num_classes)

    bert.eval()
    gnn.eval()
    fusion.eval()

    with torch.no_grad():
        _, t_cls, h_text = bert(tokens["input_ids"], tokens["attention_mask"])
        _, g = gnn(graph.x, graph.edge_index, batch_idx)
        logits, emotion, _ = fusion(g, t_cls, h_text)

        probs = torch.sigmoid(logits).squeeze(0).numpy()
        valence, arousal = emotion.squeeze(0).numpy()

    top3 = probs.argsort()[-3:][::-1]

    print("=== Demo Inference Output ===")
    print(f"Query: {caption_text}")
    print(f"Valence: {valence:.3f} | Arousal: {arousal:.3f}")
    print(f"Top 3 Tags: {top3.tolist()} | Scores: {probs[top3].round(3).tolist()}")

if __name__ == "__main__":
    sample_caption = "A fast electronic synthesizer track with driving drum beats and upbeat energy."
    run_inference(sample_caption)
