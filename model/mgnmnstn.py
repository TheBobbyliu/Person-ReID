import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck
from model.auxillary.MobileNetV2 import MobileNetV2, InvertedResidual
from model.auxillary.STNModule import SpatialTransformer

import math
from copy import deepcopy
from setproctitle import setproctitle
from utils.utility import load_state_dict
setproctitle("MobileNet+SpatialTransform")

def make_model(args):
    return MGN(args)

class MGN(nn.Module):
    def __init__(self, args):
        super(MGN, self).__init__()
        num_classes = args.num_classes

        mobilenet = MobileNetV2(1000)
        state_dict = torch.load('./model/mobilenet_v2.pth.tar')
        mobilenet.load_state_dict(state_dict)
        self.base_params = mobilenet.parameters()

        self.STN = SpatialTransformer(3, (args.height, args.width), 3)

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
        
        if args.pool == 'max':
            pool2d = nn.MaxPool2d
        elif args.pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise Exception()

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))

        reduction = nn.Sequential(nn.Conv2d(1280, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(args.feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(args.feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

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
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        x, b = self.STN(x)
        x = self.backone(x)

        # 12, 4
        p1 = self.p1(x)
        # 24, 8
        p2 = self.p2(x)
        p3 = self.p3(x)
        # TODO:: add MobileNetV2 to the model, squeeze feature from 384, 128 --> 24, 8 (16times, 4)

        # 1, 1
        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        # 2, 1
        zp2 = self.maxpool_zp2(p2)
        # 1, 1
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        # 3, 1
        zp3 = self.maxpool_zp3(p3)
        # 1, 1
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        
        # reduce dimensions to args.feats
        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)
        
        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3

        


