import torch.nn as nn
import math
import torch
import math
from copy import deepcopy
from torchvision.models.resnet import resnet50
from time import time

"""
model: AMG
reference: Ning Liu,et al. 
ADCrowdNet: An Attention-injective Deformable Convolutional Network for Crowd Understanding
"""
# first try:: add AMG to Resnet

def make_model(args):
    return AMG(args)

# for every layer, using different dilation rate to change scales of kernel
# we can check batchnorm's effect by adding or removing it
# batchnorm could be added to end of every conv or after concat
# differ from resnet, we use shrink_rate to reduce memory use
# we can also test invert residual's effect in this process by adding group to conv and a 1*1 conv in the end.
class multidilation(nn.Module):
    def __init__(self, inp, oup, shrink_rate = 2, stride = 1, down=False):
        super(multidilation, self).__init__()
        hidden_dim = inp // shrink_rate
        self.down = down
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, (1,1)),
            nn.BatchNorm2d(hidden_dim, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True)
        )

        if self.down:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp, oup, (1,1), padding=0, stride=stride),
                nn.BatchNorm2d(oup, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True)
            )

        self.conv2_1 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, dilation=1, padding=1)
        self.conv2_2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, dilation=3, padding=3)
        self.conv2_3 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, dilation=6, padding=6)            
        self.conv2_4 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, dilation=9, padding=9)

        self.conv_out = nn.Sequential(
            # pw-linear
            nn.BatchNorm2d(hidden_dim * 4, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim * 4, oup, 1, stride=1, bias=False),
            nn.BatchNorm2d(oup, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True),            
        )

        self._initialize_weights()

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2_1(x0)
        x2 = self.conv2_2(x0)
        x3 = self.conv2_3(x0)
        x4 = self.conv2_4(x0)
        x_out = torch.cat((x1, x2, x3, x4), 1)
        x_out = self.conv_out(x_out)
        if self.down:
            return x_out + self.downsample(x)
        else:
            return x_out + x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1., std=0.02)
                nn.init.constant_(m.bias, 0.)
        

class AMG(nn.Module):
    def __init__(self, args):
        super(AMG, self).__init__()
        num_classes = args.num_classes
        self.args = args

        resnet = resnet50(pretrained=True)
        self.base_params = resnet.parameters()
        
        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        # 48,16
        # for multidilation net, it is better to resize pedestrian's images to (384,384) or (256,256)
        self.backtwo = resnet.layer2
        self.multi_layer3 = nn.Sequential(
            # corresponding to resnet.layer3
            multidilation(512, 1024, 2, 2, True),
            multidilation(1024, 1024, 4, 1),
            multidilation(1024, 1024, 4, 1),
            multidilation(1024, 1024, 4, 1),
            multidilation(1024, 1024, 4, 1),
        )
        self.multi_layer4 = nn.Sequential(
            # corresponding to resnet.layer4
            multidilation(1024, 2048, 2, 2, True),
            multidilation(2048, 2048, 4, 1),
            multidilation(2048, 2048, 4, 1)
        )
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Conv2d(2048, num_classes, 1, 1)
        self._initialize_fc(self.fc)
        """
        # 12, 4 --> 1, 1
        self.p1 = nn.Sequential(
            higher(128, 256),
            higher(256, 256, 1),
            higher(256, 256),
            higher(256, 512),
            nn.Conv2d(512, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum = 0.1, affine=True, track_running_stats= True),
            nn.ReLU6()
        )

        self.p2 = deepcopy(self.p1)
        self.p3 = deepcopy(self.p1)

        self.avgpool_p2 = nn.AvgPool2d((3, 4), 1)
        self.avgpool_p3 = nn.AvgPool2d((4, 4), 1)
        
        reduction = nn.Sequential(nn.Conv2d(1024, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU())
        self.reduction_module = nn.ModuleList()

        self.fc_module = nn.ModuleList()

        for i in range(self.args.slice_p2+self.args.slice_p3+3):
            self.reduction_module.append(deepcopy(reduction))

        for i in range(self.args.slice_p2+self.args.slice_p3+3):
            fc = nn.Conv2d(args.feats, num_classes, 1)
            self.fc_module.append(fc)

        self._initialize_weights()
        load_state_dict(mobilenet, torch.load('./model/mobilenet_v2.pth.tar'), args)
        """

    def forward(self, x):
        t1 = time()
        x = self.backone(x)
        t2 = time()
        x = self.backtwo(x)
        t3 = time()
        x = self.multi_layer3(x)
        t4 = time()
        x = self.multi_layer4(x)
        t5 = time()
        #print('one: %.4f two: %.4f three: %.4f four: %.4f'%(t2-t1,t3-t2,t4-t3,t5-t4))
        feat = self.adaptiveAvgPool(x)
        cls = self.fc(feat)
        cls = cls.squeeze(3).squeeze(2)
        feat = feat.squeeze(3).squeeze(2)
        
        return feat, cls
        """
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        slice_result = []
        slice_result.append(self.adaptiveAvgPool(p1))
        slice_result.append(self.adaptiveAvgPool(p2))
        slice_result.append(self.adaptiveAvgPool(p3))

        zp2 = self.avgpool_p2(p2)
        slice_result.append(zp2[:, :, 0:1, :])
        slice_result.append(zp2[:, :, 1:2, :])
        slice_result.append(zp2[:, :, 2:3, :])
        slice_result.append(zp2[:, :, 3:4, :])

        zp3 = self.avgpool_p3(p3)
        slice_result.append(zp3[:, :, 0:1, :])
        slice_result.append(zp3[:, :, 1:2, :])
        slice_result.append(zp3[:, :, 2:3, :])
        
        reduce_result = []
        class_outputs = []
        for i in range(len(self.reduction_module)):
            reduce_result.append(self.reduction_module[i](slice_result[i]))
            class_outputs.append(self.fc_module[i](reduce_result[i]).squeeze(dim=3).squeeze(dim=2))

        feats = torch.cat(reduce_result, 1).squeeze(dim=3).squeeze(dim=2)

        return feats, reduce_result[0].squeeze(dim=3).squeeze(dim=2), reduce_result[1].squeeze(dim=3).squeeze(dim=2), reduce_result[2].squeeze(dim=3).squeeze(dim=2), class_outputs
        """
    def _initialize_fc(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
        if m.bias is not None:
            m.bias.data.zero_()