"""
    CIS 6200 -- Deep Learnig Final Project
    Extract frames from habitat sim
    April 2024
"""

import numpy as np
from scipy.spatial.transform import Rotation
from lib.matcher import Matcher
from lib.habitat_wrapper import HabitatWrapper

class ImageExtractor:

    def __init__(self, file_path):
        self.matcher_ = Matcher()
        self.sim_wrapper_ = HabitatWrapper(file_path)
        self.file_path_ = file_path

    def get_pose_trace(self, generic_path, subguide):
        instr_id = str(subguide['instruction_id'])
        data_path = self.file_path_ + \
            "pose_traces/rxr_train/{:06}_guide_pose_trace.npz".format(instr_id)
        pose_trace = np.load(data_path)

    def calculate_rotation(self, pose1, pose2):
        # y is zero, angle increases looking right, z is up
        #first calculate the heading 
        x_diff = pose1[0] - pose2[0]
        y_diff = pose1[1] - pose2[1]
        heading = np.arctan2(y_diff, x_diff)
        
        # convert to habitats convention
        if heading <= (np.pi/2):
            heading_habitat_frame = np.pi/2 - heading
        elif heading > (np.pi/2) and heading <= np.pi:
            heading_habitat_frame = (3*np.pi)/2 + (np.pi/2)-heading-(np.pi/2)
        elif heading > np.pi and heading <= (3*np.pi)/2:
            heading_habitat_frame = np.pi + ((3*np.pi/2) - heading)
        else:
            heading_habitat_frame = np.pi/2 + (2*np.pi - heading)

        # make a rotation matrix
        rot = Rotation.from_euler('z', heading_habitat_frame)
        
        return rot 


    def get_frame_from_sim(self, extrinsics, instruction_ids):
        # get unique extrinsics
        self.matcher_.match(instruction_ids)
        pose_matrices = self.matcher_.poses_from_match(extrinsics)

        # parse rotations and poses
        rotations = pose_matrices[:3,:3,:]
        poses = pose_matrices[:3,-1,:]

        # you have n roations and poses where n is the number of waypoints
        # use the poses to calculate yaw


    def get_image(self, generic_path, subguide):
        instr_id = str(subguide['instruction_id'])
        scene_id = str(subguide['scene_id'])



