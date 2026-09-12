import torch

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss
