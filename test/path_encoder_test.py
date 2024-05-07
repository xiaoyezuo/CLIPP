"""
    CIS 6200 Final Project
    Unit test to test path encoders and decoders from the dataset. 
    April 2024
"""

import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from models.pose_encoder_small import PoseEncoder
from lib.pose_extractor import Interpolator

interpolator = Interpolator(96)
test_path = [[1,1,1],[2,2,2],[3,3,3]]
test_path = interpolator.interpolate(test_path)
test_path = torch.from_numpy(np.array(test_path)).to(torch.float32)

path_encoder = PoseEncoder(96, 512, 512)
path_encoder.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models/encoder_epoch9.pth')))
path_encoder.eval()
# path_encoder = torch.load(os.path.join(PROJECT_ROOT, 'models/encoder_epoch9.pth'))
# path_decoder = torch.load(os.path.join(PROJECT_ROOT, 'models/decoder_epoch9.pth'))

# path_encoder.eval()
# path_decoder.eval()

# print(path_encoder.parameters())
encoded_path = path_encoder(test_path)
print(encoded_path)