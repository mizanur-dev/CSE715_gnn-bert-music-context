import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau

    def forward(self, g, t):
        g_norm = F.normalize(g, p=2, dim=-1)
        t_norm = F.normalize(t, p=2, dim=-1)
        logits = torch.matmul(g_norm, t_norm.t()) / self.tau
        targets = torch.arange(g.size(0), device=g.device)
        loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) / 2.0
        return loss

def evaluate_retrieval(g, t, ks=[1, 5, 10]):
    g_norm = F.normalize(g, p=2, dim=-1)
    t_norm = F.normalize(t, p=2, dim=-1)
    sim = torch.matmul(t_norm, g_norm.t())
    n = t_norm.size(0)
    recalls = {}
    for k in ks:
        pred_topk = torch.topk(sim, k=min(k, n), dim=1).indices
        correct = (pred_topk == torch.arange(n, device=sim.device).unsqueeze(1)).any(dim=1)
        recalls[f"R@{k}"] = float(correct.float().mean().item())
    return recalls
