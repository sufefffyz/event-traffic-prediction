import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data
        return hidden
