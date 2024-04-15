"""
    CIS 6200 - Deep Learning
    Autoencoder model for Pose Encoding 
    April 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseEncoder(nn.Module):

    def __init__(self):
        super.__init__()

    def forward(self, x):
        return x
