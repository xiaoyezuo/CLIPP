"""
    CIS 6200 -- Deep Learning Final Project
    Matcher class for uids to extrinsics
    April 2024
"""

import numpy as np

class Matcher:

    def __init__(self):
        self.uids_ = None

    def match(self, ids):
        ids, uids = np.unique(ids, return_index=True)
        self.uids_ = np.sort(uids)

    def poses_from_match(self, poses):
        return poses[self.uids_]
