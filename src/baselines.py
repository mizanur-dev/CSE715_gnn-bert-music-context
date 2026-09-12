import torch
import torch.nn as nn
import torch.nn.functional as F

class MajorityRandomBaseline:
    def __init__(self, num_classes=50, prior_prob=0.1):
        self.num_classes = num_classes
        self.prior_prob = prior_prob

    def predict(self, batch_size):
        return torch.full((batch_size, self.num_classes), self.prior_prob)

class AudioCNNBaseline(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)
