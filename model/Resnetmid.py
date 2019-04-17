import torch.nn as nn
import math
import torch
import math
from copy import deepcopy
from torchvision.models.resnet import resnet50
from time import time

"""
model: Resnet with mid-level feature fusion
reference: Qian Yu, et al. 
The Devil is in the Middle: Exploiting Mid-level Representations for
Cross-Domain Instance Matching
"""
# zero try:: Resnetmid

def make_model(args):
    return Resnetmid(args)

class Resnetmid(nn.Module):
    def __init__(self, args):
        super(Resnetmid, self).__init__()
        num_classes = args.num_classes
        self.args = args

        resnet = resnet50(pretrained=True)
        self.base_params = nn.Sequential(*list(resnet.children())[:-3]).parameters()
        
        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.final1 = list(resnet.layer4.children())[0]
        self.final2 = list(resnet.layer4.children())[1]
        self.final3 = list(resnet.layer4.children())[2]
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_module = nn.ModuleList()
        fc = nn.Linear(2048, num_classes)
        self._initialize_fc(fc)
        for i in range(3):
            self.fc_module.append(deepcopy(fc))

        self.mid_fc = nn.Linear(4096, 2048)
        self.final_fc = nn.Linear(4096, num_classes)
        self._initialize_fc(self.mid_fc)
        self._initialize_fc(self.final_fc)
        
    def forward(self, x):
        x = self.backone(x)
        x1 = self.final1(x)
        x2 = self.final2(x1)
        x3 = self.final3(x2)
        feat1 = self.adaptiveAvgPool(x1).squeeze(3).squeeze(2)
        feat2 = self.adaptiveAvgPool(x2).squeeze(3).squeeze(2)
        feat3 = self.adaptiveAvgPool(x3).squeeze(3).squeeze(2)
        feat_mid = torch.cat((feat1, feat2), 1)
        feat_mid = self.mid_fc(feat_mid)
        cls1 = self.fc_module[0](feat1)
        cls2 = self.fc_module[1](feat2)
        cls3 = self.fc_module[2](feat3)
        feature = torch.cat((feat_mid, feat3), 1)
        cls4 = self.final_fc(feature)
        # output feature b*4096
        return feature, [cls1, cls2, cls3, cls4]
        
    def _initialize_fc(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            m.bias.data.zero_()
