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
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from skimage.measure import regionprops
import random



class MaskRCNNDataset(Dataset):
    def __init__(self, path, img_folder="for_maskrcnn_image", transforms=None):
        self.path = path
        self.img_folder = os.path.join(path, img_folder)
        self.file_names = [file_name.split(".")[0] for file_name in os.listdir(self.img_folder)]
        self.transforms = transforms

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        try :
            image = Image.open(os.path.join(self.img_folder, file_name + ".jpg")).convert('RGB')
        except :
            image = Image.open(os.path.join(self.img_folder, file_name + ".jpg.jpg")).convert('RGB')

        mask = np.load(os.path.join(os.path.join(self.path, "masking"), file_name + ".npy"))
        bbox = regionprops(mask)[0].bbox

        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]

        target = {
            'boxes': np.array([bbox], dtype=np.float32),
            'masks': np.array([mask/255],dtype=np.uint8),
            'labels': [1],
            'area': (mask/255).sum(),
            'iscrowd': 0}
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.uint8)            

        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(
                image, target)

        return image, target


class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, target):
        w, h = image.size
        image = image.resize(self.size)

        _masks = target['masks'].copy()
        masks = np.zeros((_masks.shape[0], self.size[0], self.size[1]))
        
        for i, v in enumerate(_masks):
            v = Image.fromarray(v).resize(self.size, resample=Image.BILINEAR)
            masks[i] = np.array(v, dtype=np.uint8)

        target['masks'] = masks
        target['boxes'][:, [0, 2]] *= self.size[0] / w
        target['boxes'][:, [1, 3]] *= self.size[1] / h
        
        return image, target
        

class RandomRotate:
    def __init__(self, size, angles=[-75, -60, -30, 0, 30, 60, 75]):
        self.angles = angles
        self.size = size
        
    def __call__(self, image, target):

        angle = random.choice(self.angles)

        image = TF.rotate(image, angle)


        _masks = target['masks'].copy()
        masks = np.zeros((_masks.shape[0], self.size[0], self.size[1]))
        bboxs = np.zeros((_masks.shape[0], 4))
        for i, v in enumerate(_masks):
            v = Image.fromarray(v)
            mask = np.array(TF.rotate(v, angle=angle), dtype=np.uint8)
            masks[i] = mask


        target['masks'] = masks
        target['boxes'] = bboxs

        
        return image, target


class RandomFlip:
    def __init__(self, size, prob=0.25):
        self.prob = prob
        self.size = size

    def __call__(self, image, target):
        flag = random.uniform(0, 1)
        v_flag = random.choice([True, False])

        _masks = target['masks'].copy()
        masks = np.zeros((_masks.shape[0], self.size[0], self.size[1]))
        bboxs = np.zeros((_masks.shape[0], 4))

        if flag > self.prob:
            if v_flag :
                image = TF.vflip(image)
                for i, v in enumerate(_masks):
                    v = Image.fromarray(v)
                    mask = np.array(TF.vflip(v), dtype=np.uint8)
                    masks[i] = mask

            else :
                image = TF.hflip(image)
                for i, v in enumerate(_masks):
                    v = Image.fromarray(v)
                    mask = np.array(TF.hflip(v), dtype=np.uint8)
                    masks[i] = mask


            target['masks'] = masks
            target['boxes'] = bboxs
        return image, target

class ReBBOX:
    def __call__(self, image, target):

        bboxs = np.zeros((target['masks'].shape[0], 4))
        _masks = target['masks'].copy()
        for i, v in enumerate(_masks):
            v = v.astype(np.uint8)
            bbox = regionprops(v)[0].bbox
            bboxs[i, 0] = bbox[1]
            bboxs[i, 1] = bbox[0]
            bboxs[i, 2] = bbox[3]
            bboxs[i, 3] = bbox[2]
        target['boxes'] = bboxs
        return image, target
        


class ToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        
        return image, target
