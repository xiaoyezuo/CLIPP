"""
    CIS 6200 Final Project
    Extract poses from paths for autoencoder
    April 2024
"""

import numpy as np
import gzip
import json


class PoseExtractor:

    def __init__(self, path):

        self.pose_ = None
        self.train_guide_ = list()

        self.load(path)

    @property
    def pose(self):
        return self.pose_

    @pose.setter
    def pose(self, p):
        self.pose_ = p

    def load(self, path):
        with gzip.open(path, 'r') as f:
            print("[PoseExtractor] found file %s, loading..." %path)
            self.train_guide_ = [json.loads(line) for line in f]

    def path_from_guide(self):
        # here we can get the path of UIDs from the train guide
        # but then where to we get the actual poses of the points on the path.
        pass

