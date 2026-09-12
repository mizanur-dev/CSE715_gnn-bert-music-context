import torch
from sklearn.metrics import f1_score

def evaluate_metrics(y_true, y_pred):
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return {'Macro-F1': macro, 'Micro-F1': micro}
