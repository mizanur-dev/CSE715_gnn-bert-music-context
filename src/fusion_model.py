import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, g_dim=128, t_dim=768, hidden_dim=128):
        super().__init__()
        self.w_q = nn.Linear(g_dim, hidden_dim)
        self.w_k = nn.Linear(t_dim, hidden_dim)
        self.w_v = nn.Linear(t_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, g, h_text):
        q = self.w_q(g).unsqueeze(1)
        k = self.w_k(h_text)
        v = self.w_v(h_text)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, v).squeeze(1)
        return context, attn_weights

class GNNBERTFusion(nn.Module):
    def __init__(self, g_dim=128, t_dim=768, num_classes=50, mode="cross_attention"):
        super().__init__()
        self.mode = mode
        self.cross_attn = CrossAttention(g_dim, t_dim, 128)

        if mode == "cross_attention":
            fused_dim = g_dim + 128
        elif mode == "early_concat":
            fused_dim = g_dim + t_dim
        elif mode == "gnn_only":
            fused_dim = g_dim
        elif mode == "bert_only":
            fused_dim = t_dim

        self.tag_head = nn.Linear(fused_dim, num_classes)
        self.emotion_head = nn.Linear(fused_dim, 2)

    def forward(self, g, t_cls, h_text):
        if self.mode == "cross_attention":
            context, _ = self.cross_attn(g, h_text)
            z = torch.cat([g, context], dim=-1)
        elif self.mode == "early_concat":
            z = torch.cat([g, t_cls], dim=-1)
        elif self.mode == "gnn_only":
            z = g
        elif self.mode == "bert_only":
            z = t_cls

        tag_logits = self.tag_head(z)
        emotion_preds = self.emotion_head(z)
        return tag_logits, emotion_preds, z
