import torch
import torch.nn as nn
from transformers import AutoModel

class MusicBERTClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=50):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_dim = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        h_text = outputs.last_hidden_state
        cls_token = h_text[:, 0, :]
        logits = self.classifier(cls_token)
        return logits, cls_token, h_text
