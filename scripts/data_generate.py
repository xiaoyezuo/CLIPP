import os, sys
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json, gzip, pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from lib.dataset import TextImagePathDataset

pose_path = "/home/zuoxy/VLA-Nav/data/pose/pose_seq.json"
rxr_guide_path =  "/home/zuoxy/ceph_old/rxr-data/rxr_train_guide.jsonl.gz"
img_dir = "/home/zuoxy/VLA-Nav/data/img/"

data_save_dir = "/home/zuoxy/VLA-Nav/data/"

with open(pose_path, 'r') as f:
    pose_all = json.load(f)

with gzip.open(rxr_guide_path, 'r') as f:
        train_guide_data = [json.loads(line) for line in f]

images = []
poses = []
texts = []
for data in train_guide_data:
    instr_id = data['instruction_id']
    if(os.path.exists(img_dir + "{:06}".format(int(instr_id)))):
        print(f"Processing instruction: {instr_id}")
        images.append(img_dir + "{:06}/000000.png".format(int(instr_id)))
        texts.append(data['instruction'])
        poses.append(pose_all[str(instr_id)])

with open(data_save_dir + "data.json", 'w') as f:
    json.dump([images, texts, poses], f)
        