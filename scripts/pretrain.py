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

from lib.pose_extractor import PoseExtractor
from lib.pose_encoder import PoseEncoder
from lib.pose_decoder import PoseDecoder

PATH = "/raid0/docker-raid/jasonah/VLA-Nav-Data/rxr-data/rxr_train_guide.jsonl.gz"

def main():
    pe = PoseExtractor(PATH)
    

if __name__ == "__main__":
    main()
