"""
    CIS 6200 -- Deep Learning Final Project
    Object to organize input data
    April 2024
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, ViTModel
from PIL import Image
import torch

class TextImagePathDataset(Dataset):

    def __init__(self, images_paths, text_data, pose_data=None):

        self.img_paths = images_paths
        self.text_data = text_data
        # self.poses_ = pose_data
        self.img_tokenizer = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.text_toeknizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.img_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
        # self.linear1 = torch.nn.Linear(768, 512)
        # self.linear2 = torch.nn.Linear(768, 512)

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        #process text
        text = self.text_data[idx]
        text_inputs = self.text_toeknizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.text_model(**text_inputs)
            last_hidden_states = outputs.last_hidden_state
            text_embedding = last_hidden_states[:, 0, :]
            # text_embedding = self.linear2(text_embedding)

        #process image 
        image_paths = self.img_paths[idx]
        images = [Image.open(image_path) for image_path in image_paths]
        img_inputs = self.img_tokenizer(images=images, return_tensors="pt")
        with torch.no_grad():
            outputs = self.img_model(**img_inputs)
            last_hidden_states = outputs.last_hidden_state
            img_embedding = last_hidden_states[:, 0, :]
            # img_embedding = self.linear1(img_embedding)

        return text_embedding, img_embedding
