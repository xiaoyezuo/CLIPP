"""
    CIS 6200 -- Deep Learning Final Project
    Unit test for the habitat frame extractor
    April 2024
"""

import os 
import sys
import gzip
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from lib.frame_extractor import ImageExtractor

def main(path):
    im = ImageExtractor(path)

    with gzip.open(path+"rxr_train_guide.jsonl.gz") as f:
        print("found file %s, loading..." %path)
        for line in f:
            guide = [json.loads(line)]
            break

    for subguide in guide:
        print(subguide)
        rgb = im.get_image( subguide )
        print("rgb: ", len(rgb))
        break

if __name__ == "__main__":

    path = "/home/jasonah/data/VLA-Nav-Data/rxr-data/" 
    main( path )

