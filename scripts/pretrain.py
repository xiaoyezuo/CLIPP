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

def main():
    pass

if __name__ == "__main__":
    main()
