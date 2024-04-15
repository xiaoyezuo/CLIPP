"""
    CIS 6200 - Deep Learning
    Autoencoder model for Pose Encoding 
    April 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    NOTE:
    Notice in_dim is the dimension of our original input so it is
    the dimension we want the decoder to output, hence layer3 of the 
    decoder outputs in_dim.
"""

class PoseEncoder(nn.Module):
    # encode poses of path, by compressing them to out_dim
    def __init__(self, in_dim, hidden, out_dim):
        super.__init__()
        # layers
        self.layer1_ = nn.Linear(in_dim, hidden)
        self.layer2_ = nn.Linear(hidden, hidden)
        self.layer3_ = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # forward pass
        out = F.relu(self.layer1_(x))
        out = F.relu(self.layer2_(out))
        out = F.relu(self.layer3_(out))
        return out

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
