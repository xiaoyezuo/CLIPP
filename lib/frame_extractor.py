"""
    CIS 6200 Final Project
    Extract frames from habitat sim
    April 2024
"""

import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from lib.matcher import Matcher

class ImageExtractor:

    def __init__(self):
        self.sim_ = None
        self.matcher_ = Matcher()

    def __del__(self):
        self.sim_.close()

    def get_pose_trace(self, generic_path, subguide):
        instr_id = str(subguide['instruction_id'])
        data_path = generic_path + \
            "pose_traces/rxr_train/{:06}_guide_pose_trace.npz".format(instr_id)
        pose_trace = np.load(data_path)            


    def get_frames(self, extrinsics, instruction_ids):
        # get unique extrinsics
        self.matcher_.match(instruction_ids)
        pose_matrices = self.matcher_.poses_from_match(extrinsics)

        # parse rotations and poses
        rotations = pose_matrices[:3,:3,:]
        poses = pose_matrices[:3,-1,:]

        # you have n roations and poses where n is the number of waypoints
        # use the poses to calculate yaw

    def place_agent(self, ):
        state = habitat_sim.AgentState()
        state.position 

    def update_sim(self, scene_id):
        self.sim_ = habitat_sim.Simulator(self.make_config(scene_id))

    def make_config(self, generic_path, scene_id):
        scene_file = generic_path + "/data/mp3d/%s/%s.glb" %(scene_id, scene_id)

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_file

        camera_resolution = [240, 320]
        sensors = {
            "rgba_camera": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": camera_resolution,
                "position": [0.0, 0.0, 0.0],  # ::: fix y to be 0 later
            },
            "semantic_camera": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": camera_resolution,
                "position": [0.0, 0.0, 0.0],
            },
        }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_specs.append(sensor_spec)

        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        return habitat_sim.Configuration(backend_cfg, [agent_cfg])


    def get_image(self, generic_path, subguide):
        instr_id = str(subguide['instruction_id'])
        img_path = generic_path + instr_id + "/"



