"""Small independently implemented PyTorch models; no pretrained weights."""
from torch import nn


def make_model(architecture):
    if architecture == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    if architecture == "cnn":
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )
    raise ValueError(architecture)
