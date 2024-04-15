"""
    CIS 6200 Final Project
    Extract poses from paths for autoencoder
    April 2024
"""

import numpy as np

class PoseExtractor:

    def __init__(self):

        self.pose_ = None

    @property
    def pose(self):
        return self.pose_

    @pose.setter
    def pose(self, p):
        self.pose_ = p
