"""
    CIS 6200 Final Project
    Extract poses from paths for autoencoder
    April 2024
"""

import numpy as np
import gzip
import json

class Matcher:

    def __init__(self):
        self.uids_ = None


    def match(self, ids):
        ids, uids = np.unique(ids, return_index=True)
        self.uids_ = np.sort(uids)


    def poses_from_match(self, poses):
        return poses[self.uids_]


class PoseExtractor:

    def __init__(self, path):

        self.pose_ = None
        self.train_guide_ = list()
        self.generic_path_ = path.split("rxr_train")[0]

        self.matcher_ = Matcher()

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
            print("[PoseExtractor] guide loaded")        


    def get_path_poses(self, pose_path):
        pose_trace = np.load(pose_path)
        poses = pose_trace["extrinsic_matrix"][:,:-1,-1]
        
        self.matcher_.match(pose_trace["pano"])
        unique_poses = self.matcher_.poses_from_match(poses)
        return unique_poses


    def path_from_guide(self):
        # here we can get the path of UIDs from the train guide
        # but then where to we get the actual poses of the points on the path.
        assert len(self.train_guide_) > 0, "[PoseExtractor] Training Guide is not loaded."

        for guide in self.train_guide_:
            print(guide)
            instruction_id = guide["instruction_id"]
            pose_path = self.generic_path_+ \
                "pose_traces/rxr_train/{:06}_guide_pose_trace.npz".format(instruction_id)
            
            unique_poses = self.get_path_poses(pose_path)
            #break

