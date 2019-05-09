import torch.nn as nn
import math
import torch
import math
from copy import deepcopy
from torchvision.models.resnet import resnet50
from time import time

def make_model(args):
    return Resnet(args)

class Resnet(nn.Module):
    def __init__(self, args):
        super(Resnet, self).__init__()
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
            resnet.layer4
        )
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
        self._initialize_fc(self.fc)

    def forward(self, x):
        x = self.backone(x)
        #print('one: %.4f two: %.4f three: %.4f four: %.4f'%(t2-t1,t3-t2,t4-t3,t5-t4))
        feat = self.adaptiveAvgPool(x).squeeze(3).squeeze(2)
        cls = self.fc(feat)
        return feat, cls

    def _initialize_fc(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            m.bias.data.zero_()
