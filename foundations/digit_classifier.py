import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.layer1=nn.Linear(784,512)
        self.layer2=nn.Linear(512,10)
        self.dropout=nn.Dropout(0.2)

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        torch.manual_seed(0)
        x1= F.relu(self.layer1(images))
        x1=self.dropout(x1)
        x2=torch.sigmoid(self.layer2(x1))
        return torch.round(x2,decimals=4)
