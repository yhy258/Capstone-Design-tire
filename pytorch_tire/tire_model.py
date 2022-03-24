import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



def make_maskrcnn(PATH, device):
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
    model.load_state_dict(torch.load(PATH, map_location=device))
    
    return model

    
class GradLayer(nn.Module):
    
    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x[0].shape[0] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x



"""
    Mask RCNN 사용 대신 Rule based 형태로, input이 특정 영역에 대해 타이어 위치를 통제한 상태로 들어오니까
    해당 위치를 고정적인 추가 정보로 사용하는게 나을 수 있을 것 같음.
    일단 코드는 maskrcnn.
"""
class tire_model(nn.Module) :
    def __init__(self, path, device):
        super().__init__()
        
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        # mobilenet.features[0][0] = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        """make feature extractor"""
        
        self.feature_extractor = mobilenet.features
        
        # for i, child in enumerate(self.feature_extractor.children()):
        #     if i < 3 :
        #         for param in child.parameters():
        #             param.requires_grad = False
        
        """Mask RCNN"""
        self.maskrcnn = make_maskrcnn(path, device)
        for param in self.maskrcnn.parameters():
            param.requires_grad = False
        self.maskrcnn.eval()
        
        # self.sobel_filter = GradLayer() # edge image를 어떻게 사용할지 생각해보자.
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.last = nn.Sequential(
                    nn.Linear(in_features=577, out_features=128, bias=True),
                    nn.Hardswish(),
                    nn.Linear(in_features=128, out_features=1, bias=True),
                    nn.Sigmoid()
                    )
        
        
    """
        eval, train 의 차이는 Dropout, BatchNorm 등에서 일어난다. nn.Module을 상속받은 모듈에만 적용 가능.
    """
    
    def set_train(self):
        self.feature_extractor.train()
        
    def set_eval(self):
        self.feature_extractor.eval()     
        
    def forward(self, x):
        bs = x.size(0)
        # edge = self.sobel_filter(x)

        mask = torch.stack([m['masks'][0] for m in self.maskrcnn(x)], dim=0) # bs, 1, h, w
        
        feat_mask = F.interpolate(mask, size=(7, 7)) # bs, 1, 7, 7
        
        # x = torch.cat([x, edge], dim=1)
        feat = self.feature_extractor(x)  # bs, c, 7, 7
        feat = torch.cat([feat_mask, feat], dim=1) # bs, c+1, 7 ,7
        
        
        feat = self.last(self.avgpool(feat).view(bs,-1))
        return feat
        