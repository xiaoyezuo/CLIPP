"""
    CIS 6200 Final Project
    Extract frames from habitat sim
    April 2024
"""

import numpy as np

class ImageExtractor:

    def get_image_metadata(self, generic_path, subguide): 
        instr_id = str(subguide['instruction_id'])
        img_path = generic_path + instr_id + "/"

    def get_pose_trace(self, generic_path, subguide):
        instr_id = str(subguide['instruction_id'])
        data_path = generic_path + \
            "pose_traces/rxr_train/{:06}_guide_pose_trace.npz".format(instr_id)
        pose_trace = np.load(data_path)            

    def get_image(self, generic_path, subguide):
        instr_id = str(subguide['instruction_id'])
        img_path = generic_path + instr_id + "/"



