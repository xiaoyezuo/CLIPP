"""
    CIS 6200 -- Deep Learning Final Project
    Base Extractor class
    April 2024
"""

import json
import gzip

from lib.pose_extractor import PoseExtractor
from lib.frame_extractor import ImageExtractor
from lib.text_extractor import TextExtractor
from lib.dataset import TextImagePathDataset


class BaseExtractor(PoseExtractor, ImageExtractor, TextExtractor):

    def __init__(self, path):
        super().__init__()

        self.rxr_guide_ = None
        self.generic_path_ = path.split("rxr_")[0]
        self.load(path)

    def load(self, path):
        with gzip.open(path, 'r') as f:
            print("[BaseExtractor] found file %s, loading..." %path)
            self.rxr_guide_ = [json.loads(line) for line in f]
            print("[BaseExtractor] guide loaded")

    def extract(self, idx):
        subguide = self.rxr_guide_[idx]

        text = self.get_text(subguide)
        image = self.get_text(subguide)
        path = self.get_path(self.generic_path_, subguide)
        poses = self.interpolate(path)

        return TextImagePathDataset(image, text, path)


