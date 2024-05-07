"""
    CIS 6200 -- Deep Learnig Final Project
    Extract frames from habitat sim
    April 2024
"""
import sys, os
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import gzip, json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from utils.matcher import Matcher

#extract unique viewpoint images from existing rxr images along the paths
class ImageExtractor_v2:

    def __init__(self, file_path):
        self.matcher_ = Matcher()
        self.file_path_ = file_path

    def get_pose_trace(self, instr_id):
        data_path = self.file_path_ + \
            "pose_traces/rxr_train/{:06}_guide_pose_trace.npz".format(int(instr_id))
        pose_trace = np.load(data_path)
        return pose_trace

    def calculate_rotation(self, pose1, pose2):
        # y is zero, angle increases looking right, z is up
        #first calculate the heading 
        x_diff = pose1[0] - pose2[0]
        y_diff = pose1[1] - pose2[1]
        heading = np.arctan2(y_diff, x_diff)
        #print(heading)
        if heading < 0:
            # make it positive
            heading += 2*np.pi
        
        # convert to habitats convention
        if heading <= (np.pi/2):
            heading_habitat_frame = np.pi/2 - heading + np.pi
        elif heading > (np.pi/2) and heading <= np.pi:
            heading_habitat_frame = (3*np.pi)/2 + (np.pi/2)-heading-(np.pi/2) + np.pi
        elif heading > np.pi and heading <= (3*np.pi)/2:
            heading_habitat_frame = np.pi + ((3*np.pi/2) - heading)
        else:
            heading_habitat_frame = np.pi/2 + (2*np.pi - heading) + np.pi

        # make a rotation matrix
        rot = Rotation.from_euler('z', [heading_habitat_frame+(np.pi/2)])
        
        return rot.as_euler('xyz') 

    def get_unique_frames(self, extrinsics, pano):
        # get unique extrinsics
        self.matcher_.match(pano)
        uids = self.matcher_.uids_
        # pose_matrices = self.matcher_.poses_from_match(extrinsics)
        
        # parse rotations and poses
        rotations = Rotation.from_matrix(extrinsics[:,:3,:3]).as_euler('xyz')
        poses = extrinsics[:,:-1,-1]

        image_uids = []
        for i in range(len(uids)-1):
            target_rotation = self.calculate_rotation(poses[uids[i]], poses[uids[i+1]])
            closest_rotation_idx = np.argmin(np.linalg.norm(rotations[uids[i]:uids[i+1]] - target_rotation, axis=1))+uids[i]
            image_uids.append(closest_rotation_idx)
        image_uids.append(uids[-1])

        return image_uids
     
    def extract_rotation(extrinsics):
        rotations = Rotation.from_matrix(extrinsics[:,:3,:3])
        return rotations.as_euler('xyz')

def extract(instr_id, input_dir, output_dir):
    extractor = ImageExtractor_v2("/home/zuoxy/ceph_old/rxr-data/")
    pose = extractor.get_pose_trace(instr_id)
    extrinsic = pose['extrinsic_matrix'][::10]
    pano = pose['pano'][::10]
    uids = extractor.get_unique_frames(extrinsic, pano)
    if not os.path.exists(output_dir + "{:06}".format(int(instr_id))):
        os.makedirs(output_dir + "{:06}".format(int(instr_id)))
    count=0
    pose_={}
    pose_sequence=[]
    for idx in uids:
        if not os.path.exists(input_dir+"{:06}/".format(int(instr_id))+"{:06}".format(idx)+".png"):
            continue
        img = Image.open(input_dir+"{:06}/".format(int(instr_id))+"{:06}".format(idx)+".png")
        filepath = output_dir + "{:06}/".format(int(instr_id)) + f"{count:06d}.png"
        img.save(filepath)
        pose_[count] = pose['extrinsic_matrix'][idx][:3, 3].tolist()
        pose_sequence.append(pose['extrinsic_matrix'][idx][:3, 3].tolist())
        count += 1
    return pose_, pose_sequence

pose_all = {}
pose_seq_all = {}
pose_path = PROJECT_ROOT+"/data/pose/pose.json"
pos_seq_path = PROJECT_ROOT+"/data/pose/pose_seq.json"
rxr_guide_path =  "/home/zuoxy/ceph_old/rxr-data/rxr_train_guide.jsonl.gz"

with gzip.open(rxr_guide_path, 'r') as f:
        train_guide_data = [json.loads(line) for line in f]

for data in train_guide_data:
    instr_id = data['instruction_id']
    print(f"Processing instruction: {instr_id}")
    pose_, pose_seq = extract(instr_id, "/home/zuoxy/ceph_old/navcon_video/rxr_clips/", "/home/zuoxy/VLA-Nav/data/img/")
    pose_all[instr_id] = pose_
    pose_seq_all[instr_id] = pose_seq
    if instr_id == 35000:
        break

with open(pose_path, 'w') as f:
    json.dump(pose_all, f)

with open(pos_seq_path, 'w') as f:
    json.dump(pose_seq_all, f)

