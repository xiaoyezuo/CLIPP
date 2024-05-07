import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Contrastive Langugae-Image-Path Pretraining
class CLIPP(nn.Module):
    def __init__(self, input_dim, path_dim, output_dim):
        super(CLIPP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.fc3 = nn.Linear(path_dim, output_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))#learned temperature parameter

    def forward(self, text_embed, image_embed, pose_embed):

        #convert to output dimension 
        text_features = self.fc1(text_embed)
        image_features = self.fc2(image_embed)
        path_features = self.fc3(pose_embed)

        #normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        path_features = F.normalize(path_features, p=2, dim=1)
        # print(text_features.shape, image_features.shape, path_features.shape)

        #cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_path_images = path_features @ image_features.T * logit_scale
        logits_per_images_path = logits_per_path_images.T
        logits_per_path_text = path_features @ text_features.T * logit_scale
        logits_per_text_path = logits_per_path_text.T
        logits_per_image_text = image_features @ text_features.T * logit_scale
        logits_per_text_image = logits_per_image_text.T

        return logits_per_text_path, logits_per_path_text, logits_per_images_path, logits_per_path_images, logits_per_image_text, logits_per_text_image
    
class CLIPPLoss(nn.Module):
    def __init__(self):
        super(CLIPPLoss, self).__init__()
    
    def forward(self, logits_per_text_path, logits_per_path_text, logits_per_images_path, logits_per_path_images, logits_per_image_text, logits_per_text_image):

        #compute loss
        loss = F.cross_entropy(logits_per_text_path, torch.arange(logits_per_text_path.shape[0]).to(device))
        loss += F.cross_entropy(logits_per_path_text, torch.arange(logits_per_path_text.shape[0]).to(device))
        loss += F.cross_entropy(logits_per_images_path, torch.arange(logits_per_images_path.shape[0]).to(device))
        loss += F.cross_entropy(logits_per_path_images, torch.arange(logits_per_path_images.shape[0]).to(device))
        loss += F.cross_entropy(logits_per_image_text, torch.arange(logits_per_image_text.shape[0]).to(device))
        loss += F.cross_entropy(logits_per_text_image, torch.arange(logits_per_text_image.shape[0]).to(device))
        return loss/6
        