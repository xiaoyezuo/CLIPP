"""
    CIS 6200 -- Deep Learning Final Project
    Object to organize input data
    April 2024
"""

import numpy as np

class InputData:

    def __init__(self, image: np.ndarray, text: str, poses: np.ndarray):

        self.img_ = image
        self.text_ = text
        self.poses_ = poses

    @property
    def image(self):
        return self.img_

    @property
    def text(self):
        return self.text_

    @property
    def poses(self):
        return self.poses_
