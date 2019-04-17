import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

from torchvision.models.resnet import resnet50, Bottleneck
from model.auxillary.MobileNetV2 import MobileNetV2, InvertedResidual
import math
from copy import deepcopy
from setproctitle import setproctitle
from utils.utility import load_state_dict
setproctitle("MobileNet+RPP")

def make_model(args):
    return MGN(args)

class MGN(nn.Module):
    def __init__(self, args):
        super(MGN, self).__init__()
        num_classes = args.num_classes
        self.args = args
        mobilenet = MobileNetV2(1000)
        state_dict = torch.load('./model/mobilenet_v2.pth.tar')
        mobilenet.load_state_dict(state_dict)

        self.backone = mobilenet.features[:8]
        # 384, 128 --> 24, 8
        res_conv4 = mobilenet.features[8:14]
        res_g_conv5 = mobilenet.features[14:]
        res_p_conv5 = nn.Sequential(
            InvertedResidual(96, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 320, 1, 6),
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True)
        )
        feature14_stride1 = InvertedResidual(96, 160, 1, 6)
        feature14_stride1.load_state_dict(mobilenet.features[14].state_dict())
        res_p_conv5 = nn.Sequential(
            feature14_stride1,
            deepcopy(mobilenet.features[15:])
        )
        
        # 24, 8 --> 12, 4, 1280
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        # 24, 8 --> 24, 8, 1280
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        # only set for RPP
        if args.module == 'RPP':
            self.mod = RPP(self.args)
        elif args.module == 'MGN':
            self.mod = Normal(self.args)

        # for testing, we don't corporate average pooling to the model
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1,1))

        reduction = nn.Sequential(nn.Conv2d(1280, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU())
        self.reduction_module = nn.ModuleList()
        self._init_reduction(reduction)

        self.fc_module = nn.ModuleList()

        for i in range(self.args.slice_p2+self.args.slice_p3+3):
            self.reduction_module.append(copy.deepcopy(reduction))

        for i in range(self.args.slice_p2+self.args.slice_p3+3):
            fc = nn.Conv2d(args.feats, num_classes, 1, bias=True)
            self._init_fc(fc)
            self.fc_module.append(fc)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_in')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)

        # 12, 4
        p1 = self.p1(x)
        # 24, 8
        p2 = self.p2(x)
        p3 = self.p3(x)
        # TODO:: add RRP to the model
        
        middle = []

        # 1, 1
        fg_p1 = self.adaptiveAvgPool(p1)
        fg_p2 = self.adaptiveAvgPool(p2)
        fg_p3 = self.adaptiveAvgPool(p3)

        middle.append(fg_p1)
        middle.append(fg_p2)
        middle.append(fg_p3)

        slice_results = self.mod(p2, p3)
        for i in range(len(slice_results)):
            middle.append(self.adaptiveAvgPool(slice_results[i]))

        # middle in form [fg_p1, fg_p2, fg_p3, z0_p2, xxx, z0_p3, xxx]
        # reduce dimensions to args.feats
        feats = []
        for i in range(len(middle)):
            fg = self.reduction_module[i](middle[i])
            feats.append(fg)

        # fully connected to dimension 'feats' so as to get softmax
        featsclass = []
        for i in range(len(feats)):
            l = self.fc_module[i](feats[i])
            featsclass.append(l.squeeze(dim=3).squeeze(dim=2))

        predict = torch.cat(feats, dim=1).squeeze(dim=3).squeeze(dim=2)
        
        return predict, \
               feats[0].squeeze(dim=3).squeeze(dim=2),\
               feats[1].squeeze(dim=3).squeeze(dim=2),\
               feats[2].squeeze(dim=3).squeeze(dim=2),\
               featsclass

        
class RPP(nn.Module):
    def __init__(self, args):
        super(RPP, self).__init__()
        self.args = args
        self.conv1x1_p2 = nn.Conv2d(1280, args.slice_p2, 1, bias = False)
        self.conv1x1_p3 = nn.Conv2d(1280, args.slice_p3, 1, bias = False)
        self.softmax = nn.Softmax(dim=1)
        self.norm_block = nn.Sequential(nn.BatchNorm2d(1280), nn.ReLU())
        
        nn.init.normal_(self.norm_block[0].weight, mean=1., std=0.02)
        init.kaiming_normal_(self.conv1x1_p2.weight.data, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv1x1_p3.weight.data, mode='fan_out', nonlinearity='relu')
    
    def forward(self, p2, p3):
        # using RRF for deviding
        # calculate softmax along channels
        logits_p2 = self.conv1x1_p2(p2)
        logits_p3 = self.conv1x1_p3(p3)
        prob_p2 = self.softmax(logits_p2)
        prob_p3 = self.softmax(logits_p3)

        slice_results = []
        for i in range(self.args.slice_p2):
            p_i = prob_p2[:, i, :, :].view(p2.size()[0], 1, p2.size()[2],p2.size()[3])
            y_i = p2 * p_i
            y_i = self.norm_block(y_i)
            slice_results.append(y_i)
            
        for i in range(self.args.slice_p3):
            p_i = prob_p3[:, i, :, :].view(p3.size()[0], 1, p3.size()[2],p3.size()[3])
            y_i = p3 * p_i
            y_i = self.norm_block(y_i)
            slice_results.append(y_i)
        
        return slice_results

class Normal(nn.Module):
    def __init__(self, args):
        super(Normal, self).__init__()
        self.args = args

    def forward(self, p2, p3):
        slice_results = []
        
        p2_height = p2.size(2)//self.args.slice_p2
        p3_height = p3.size(2)//self.args.slice_p3

        for i in range(self.args.slice_p2):
            slice_results.append(p2[:,:,i*p2_height:(i+1)*p2_height,:])
        for i in range(self.args.slice_p3):
            slice_results.append(p3[:,:,i*p3_height:(i+1)*p3_height,:])

        return slice_results
