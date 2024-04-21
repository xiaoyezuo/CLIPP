"""
    CIS 6200 Final Project
    Unit test to test path extraction from the dataset. 
    April 2024
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from lib.pose_extractor import PoseExtractor

def main(path):
    pe = PoseExtractor(path)
    pe.path_from_guide()

if __name__ == "__main__":

    path = "/home/vla-docker/data/VLA-Nav-Data/rxr-data/rxr_train_guide.jsonl.gz"
    main( path )

