import torch.nn as nn
import math
import torch
import math
from copy import deepcopy
from torchvision.models.vgg import vgg16
from time import time

def make_model(args):
    return VGG16(args)

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
        

class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        num_classes = args.num_classes
        self.args = args

        pretrained_vgg = vgg16(pretrained=True)
        self.backone = pretrained_vgg.features
        self.toFeat = nn.Sequential(
            nn.Linear(self.args.height//32*self.args.width//32*512, 4096, bias=True),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096, bias=True),
            nn.ReLU6(inplace=True)
        )
        # deleted one relu layer for consistency between feature and classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes, bias=True)
        )
        self.base_params = nn.Sequential(*list(pretrained_vgg.children())[0]).parameters()
        self._initialize_fc(self.toFeat)
        self._initialize_fc(self.classifier)

        
    def forward(self, x):
        x = self.backone(x)
        x = x.view((x.size(0),-1))
        feat = self.toFeat(x)
        cls = self.classifier(feat)
        return feat, cls
        
    def _initialize_fc(self, m):
        for mod in m:
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out')
                if mod.bias is not None:
                    mod.bias.data.zero_()
