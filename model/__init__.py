import os
from importlib import import_module

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args, ckpt):
        super(Model, self).__init__()

        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.nGPU = args.nGPU
        self.ckpt = ckpt

        module = import_module('model.' + args.model)
        self.model = module.make_model(args).to(self.device)
        if hasattr(self.model, 'base_params'):
            self.ckpt.base_params = self.model.base_params

        if not args.cpu and args.nGPU > 1:
            self.model = nn.DataParallel(
                self.model, range(args.nGPU)
            )
            
    def forward(self, x):
        return self.model(x)

    def get_model(self):
        if self.nGPU == 1:
            return self.model
        else:
            return self.model.module
