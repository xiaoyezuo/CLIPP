"""
    CIS 6200 - Deep Learning
    A pose decoder as part of the autoencoder
    pretraining for embedding the action space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseDecoder(nn.Module):
    # decode poses from path, by taking them from the 
    # encoder embedding space to input space
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        # layers
        self.layer1_ = nn.Linear(out_dim, hidden)
        self.layer2_ = nn.Linear(hidden, hidden)
        self.layer3_ = nn.Linear(hidden, in_dim)

    def forward(self, x):
        # forward pass
        out = F.relu(self.layer1_(x))
        out = F.relu(self.layer2_(out))
        out = F.relu(self.layer3_(out))
        return out
