"""
    CIS 6200 - Deep Learning
    Autoencoder driver software 
    Trains an autoencoder on pose data
    from habitat
    April 2024
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from lib.path_dataloader import PathDataset
from lib.pose_encoder import PoseEncoder
from lib.pose_decoder import PoseDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = "/home/vla-docker/data/VLA-Nav-Data/rxr-data/rxr_train_guide.jsonl.gz"
MODEL_PATH = "/home/vla-docker/models/"
def train(encoder, decoder, optimizer, dataloader, epochs):
    
    encoder.train()
    decoder.train()

    for ep in range(epochs):
        for p in dataloader:
            encoded = encoder(p)
            decoded = decoder(encoded)

            loss = criterion(decoded, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: %s with loss: %s" %(ep, loss))
        # save the model as each epoch
        torch.save(encoder.state_dict(), MODEL_PATH+"encoder_epoch%s.pth" %ep)
        torch.save(decoder.state_dict(), MODLE_PATH+"decoder_epoch%s.pth" %ep)
    

if __name__ == "__main__":
   
    input_dim = 96
    hidden_dim = 512
    output_fim = 256

    epochs = 10

    dataloader = PathDataLoader(PATH, interpolate=True, out_dim=96)

    encoder = PoseEncoder(input_dim, hidden_dim, output_dim)
    decoder = PoseDecoder(input_dim, hidden_dim, output_dim)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    print("[PRETRAIN] Starting pretraining")
    train(encoder, decoder, optimizer, dataloader, epochs)
    print("[PRETRAIN] Training finished")
