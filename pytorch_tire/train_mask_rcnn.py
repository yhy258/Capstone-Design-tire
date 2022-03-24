#%%
import os
import glob
import json
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from skimage.measure import label, regionprops
import random

from mask_rcnn_data import *


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

path = "/home/yhy258/Desktop/codes/타이어마모도데이터셋"

batch_size = 8
lr = 5e-4
max_size = 224
num_workers = 4
num_epochs = 40
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"Now Device : {device}")
classes = (
    'tire'
)
num_classes = 2


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, hidden_layer, len(classes)+1)





transforms_train = Compose([
    Resize((max_size, max_size)),
    RandomRotate((max_size, max_size)),
    RandomFlip((max_size, max_size)),
    ReBBOX(),
    ToTensor()])


def collate_fn(batch):
    return tuple(zip(*batch))


dataset = MaskRCNNDataset(path, transforms=transforms_train)
train_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, 
    num_workers=num_workers, collate_fn=collate_fn)


model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)



def train_fn():
    model.train()
    for epoch in range(1, num_epochs+1):
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())
            
            print(
                f"{epoch}, {i}, C: {losses['loss_classifier'].item():.5f}, M: {losses['loss_mask'].item():.5f}, "\
                f"B: {losses['loss_box_reg'].item():.5f}, O: {losses['loss_objectness'].item():.5f}, T: {loss.item():.5f}")
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0 :
            torch.save(model.state_dict(), f"/home/yhy258/Desktop/codes/타이어마모도데이터셋/maskrcnn_{epoch+1}.pth")
            print(f"save model {epoch}")

    
train_fn()
# %%

