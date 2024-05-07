import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataset import TextImagePathDataset
from scripts.model import CLIPP

import os, sys, json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from lib.dataset import TextImagePathDataset
from scripts.model import CLIPP, CLIPPLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_save_dir = os.path.join(PROJECT_ROOT, "data/")
log_save_dir = os.path.join(PROJECT_ROOT, "log/")
train_loss_path = os.path.join(log_save_dir, "train_loss.txt")
train_log_path = os.path.join(log_save_dir, "train_log.txt")

with open(data_save_dir + 'data.json') as f:
    image_data, text_data, pose_data = json.load(f)
    
dataset = TextImagePathDataset(image_data, text_data, pose_data)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = CLIPP(768, 512, 512).to(device)

params = list(model.parameters()) + list(dataset.text_encoder.parameters()) + list(dataset.pose_encoder.parameters()) + list(dataset.img_encoder.parameters())
optimizer = optim.Adam(params,lr=0.000001)
loss_fn = CLIPPLoss()

num_epochs = 10

for epoch in range(num_epochs):
    for text_embed, image_embed, pose_embed in data_loader:
        # print(text_embed.shape, image_embed.shape, pose_embed.shape)
        optimizer.zero_grad()
        logits_per_text_path, logits_per_path_text, logits_per_images_path, logits_per_path_images, logits_per_image_text, logits_per_text_image = model(text_embed, image_embed, pose_embed)
        # Compute loss
        loss = loss_fn(logits_per_text_path, logits_per_path_text, logits_per_images_path, logits_per_path_images, logits_per_image_text, logits_per_text_image)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()
        with open(train_loss_path, 'a') as loss_file:
            loss_file.write(f"{loss.item():.4f}\n")
        
    with open(train_log_path, 'a') as log_file:
            log_file.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}\n")
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # Add other relevant information like epoch, loss, etc.
}, os.path.join(PROJECT_ROOT, 'models/clipp_epoch'+str(epoch+1)+'.pth'))

    # model._save_to_state_dict(os.path.join(PROJECT_ROOT, 'models/clipp_epoch'+str(epoch+1)+'.pth'))
    dataset.text_encoder.save_pretrained(os.path.join(PROJECT_ROOT, 'models/text_encoder_epoch'+str(epoch+1)))
    torch.save(dataset.pose_encoder.state_dict(),os.path.join(PROJECT_ROOT, 'models/pose_encoder_epoch'+str(epoch+1)+'.pth'))
    dataset.img_encoder.save_pretrained(os.path.join(PROJECT_ROOT, 'models/img_encoder_epoch'+str(epoch+1)))
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

writer.close()
