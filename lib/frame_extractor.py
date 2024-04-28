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
        self.sim_ = HabitatWrapper(file_path)
        self.file_path_ = file_path

    def get_pose_trace(self, instr_id):
        data_path = self.file_path_ + \
            "pose_traces/rxr_train/{:06}_guide_pose_trace.npz".format(instr_id)
        pose_trace = np.load(data_path)
        
        return pose_trace

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

    def get_frames_from_sim(self, extrinsics, instruction_ids):
        # get unique extrinsics
        self.matcher_.match(instruction_ids)
        pose_matrices = self.matcher_.poses_from_match(extrinsics)

        # parse rotations and poses
        rotations = pose_matrices[:3,:3,:]
        poses = pose_matrices[:3,-1,:]

        # you have n roations and poses where n is the number of waypoints
        # use the poses to calculate yaw
        for i in range(len(poses)-1):
            rotations[i] = self.calculate_rotation(poses[i], poses[i+1])

        rgb = np.empty(self.sim.camera_res_.append(len(poses)))
        sem = np.empty(self.sim.camera_res_.append(len(poses)))

        for i in range(len(poses)):
            transform = self.sim_.place_agent(rotations[i], poses[i])
            rgb[i] = self.sim_.get_sensor_observations()["rgba_camera"]
            sem[i] = self.sim_.get_sensor_observations()["semantic_camera"]
        
        return rgb, sem
            
    def get_image(self, subguide, return_sem=False):
        instr_id = str(subguide['instruction_id'])
        scene_id = str(subguide['scan'])
        
        self.sim_.update_sim(scene_id)

        pose_trace = self.get_pose_trace(instr_id)

        rgb, sem = self.get_frames_from_sim(pose_trace["extrinsics"], instr_id)

        if return_sem:
            return rgb, sem
        else:
            return rgb

