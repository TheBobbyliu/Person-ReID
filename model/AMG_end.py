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
# second try:: add AMG to Resnet layer1 and layer2

def make_model(args):
    return AMG_end(args)

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

        self.conv2_1 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, dilation=1, padding=1, groups=hidden_dim)
        self.conv2_2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, dilation=3, padding=3, groups=hidden_dim)
        self.conv2_3 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, dilation=6, padding=6, groups=hidden_dim)            
        self.conv2_4 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, dilation=9, padding=9, groups=hidden_dim)

        self.conv_out = nn.Sequential(
            # pw-linear
            nn.BatchNorm2d(hidden_dim * 4, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim * 4, oup, 1, stride=1, bias=False),
            nn.BatchNorm2d(oup, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True)
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

class AMG_end(nn.Module):
    def __init__(self, args):
        super(AMG_end, self).__init__()
        num_classes = args.num_classes
        self.args = args

        resnet = resnet50(pretrained=True)
        self.base_params = resnet.parameters()
        
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

        self.backone_front = nn.Sequential(*list(self.backone.children())[:6])

        self.dilation = nn.Sequential(
            multidilation(512, 1024, 2, 2, True),
            multidilation(1024, 1024, 4, 1),
            multidilation(1024, 1024, 4, 1),
            multidilation(1024, 1024, 4, 1),
            multidilation(1024, 1024, 4, 1),
            multidilation(1024, 2048, 2, 2, True),
            multidilation(2048, 2048, 4, 1),
            multidilation(2048, 2048, 4, 1)
        )

        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
        self._initialize_fc(self.fc)

    def forward(self, x):
        x = self.backone_front(x)
        x = self.dilation(x)
        feat = self.adaptiveAvgPool(x).squeeze(3).squeeze(2)
        cls = self.fc(feat)
        return feat, cls

    def _initialize_fc(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            m.bias.data.zero_()
