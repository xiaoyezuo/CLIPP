"""
    CIS 6200 -- Deep Learning Final Project
    Object to organize input data
    April 2024
"""

import numpy as np
from torch.utils.data import Dataset
<<<<<<< HEAD
=======
import torchvision.transforms as transforms
>>>>>>> 50a24b0 (Cleared git cache)
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, ViTModel
from transformers import RobertaTokenizer, RobertaModel
from PIL import Image
import torch
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from models.pose_encoder_small import PoseEncoder
from lib.pose_extractor import Interpolator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextImagePathDataset(Dataset):

<<<<<<< HEAD
    def __init__(self, image_paths, text_data, pose_data=None):
=======
    def __init__(self, image_paths, text_data, pose_data):
>>>>>>> 50a24b0 (Cleared git cache)

        self.image_paths = image_paths
        self.text_data = text_data
        self.pose_data = pose_data
        self.img_tokenizer = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
<<<<<<< HEAD
        self.text_toeknizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.img_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
=======
        self.text_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.img_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
        # self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base').to(device)
        self.pose_encoder = PoseEncoder(96, 512, 512).to(device)
        self.pose_encoder.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models/encoder_epoch9.pth')))
        # self.pose_encoder.eval()
        self.interpolator = Interpolator(96)
        self.transform = transforms.ToTensor()

>>>>>>> 50a24b0 (Cleared git cache)
       
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        #process text
        text = self.text_data[idx]
<<<<<<< HEAD
        text_inputs = self.text_toeknizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.text_model(**text_inputs)
            last_hidden_states = outputs.last_hidden_state
            text_embedding = last_hidden_states[:, 0, :]

        #process image 
        image_path = self.image_paths[idx]
        image = Image.open(image_path) 
        img_inputs = self.img_tokenizer(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.img_model(**img_inputs)
            last_hidden_states = outputs.last_hidden_state
            img_embedding = last_hidden_states[:, 0, :]
=======
        text_inputs = self.text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding='max_length').to(device)
        text_outputs = self.text_encoder(**text_inputs)
        # last_hidden_states = text_outputs.last_hidden_state
        # text_embedding = last_hidden_states[:, 0, :].squeeze(0)
        text_embedding = text_outputs.pooler_output.squeeze()

        #process image 
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        img_inputs = self.img_tokenizer(images=image, return_tensors="pt", do_rescale=False).to(device)
        img_outputs = self.img_encoder(**img_inputs)
        img_embedding = img_outputs.pooler_output.squeeze()
        # last_hidden_states = img_outputs.last_hidden_state
        # img_embedding = last_hidden_states[:, 0, :].squeeze(0)
>>>>>>> 50a24b0 (Cleared git cache)

        #process pose 
        pose = self.pose_data[idx]
        pose = self.interpolator.interpolate(pose)
        pose = torch.from_numpy(np.array(pose)).to(torch.float32).to(device)
        pose_embedding = self.pose_encoder(pose)

        return text_embedding, img_embedding, pose_embedding
