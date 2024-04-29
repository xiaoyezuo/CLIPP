"""
    CIS 6200 -- Deep Learning Final Project
    Wrap the Habitat Sim for training
    April 2024
"""

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
import quaternion

class HabitatWrapper:

    def __init__(self, file_path):

        self.sim_ = None

    def __del__(self):
        self.sim_.close()

    def place_agent(self, rotation, pose):
        state = habitat_sim.AgentState()
        state.position = pose
        state.rotation = quaternion.from_rotation_matrix(rotation)
        self.sim_.agents[0].set_state(state)

        return self.sim_.agents[0].scene_node.transformation_matrix()
    
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