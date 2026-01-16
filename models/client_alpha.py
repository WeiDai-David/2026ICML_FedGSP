import torch
import torch.nn as nn

class ClientAlpha(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(M))

    def forward(self):
        return torch.softmax(self.logits, dim=0)
