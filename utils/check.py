import torch.nn as nn
import torch
import cv2
import os
import numpy as np
from torch.autograd import gradcheck
from tqdm import tqdm
"""
    This class is utilized for applying tricks to DNN for checking.
    etc., visualization of features, gradient check, and so on.
"""
class check():
    def __init__(self, model, cpu=False):
        self.model = model
        self.count = 0
        self.device='cpu' if cpu else 'cuda'

    # ==============Feature Extraction Module=================
    # ========================================================
    # get all middle features in the network
    def extract_all_feature(self, inputs, pick=None, savedir = './log/extract_all_feature'):
        feats = inputs.to(self.device)
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        for module in self.model.__dir__():
            feats = getattr(self.model, module)(feats)
            if pick == None:
                pick = np.random.choice(range(feats.size(1)), 10)
            for i in range(inputs.size(0)):
                for j in pick:
                    write_img(inputs[i], feats[i][j], os.path.join(savedir, '{}_{}_{}.jpg'.format(module, i, self.count)))
                    self.count += 1

    # get a set of images to the model or a part of the model to see how it works
    def extract_specific_feature(self, inputs, sequential, pick = None, savedir='./log/extract_specific_feature'):
        input_img = inputs.to(self.device)
        outputs = sequential(input_img)
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        if pick == None:
            pick = np.random.choice(range(outputs.size(1)), 10)
        for i, input in enumerate(inputs):
            for j in pick:
                write_img(input, outputs[i][j], os.path.join(savedir, '{}_{}_{}.jpg'.format(i, j, self.count)))
                self.count += 1
        return outputs

    def write_img(self, img, feature, savename, size=(256,128)):
        if feat == None:
            print('Meet None Feature, no images are written.')
            return
        feat = feature
        feat = feat.cpu().data.numpy()
        feat = cv2.normalize(feat, None, alpha=60, beta = 180, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        feat = np.repeat(feat.reshape((feat.shape[0],feat.shape[1],1)),3, axis=2)
        feat = cv2.resize(feat, size)
        feat[:,:,0] += 30
        feat[:,:,1] -= 30
        feat = np.uint8(feat)
        feat = cv2.applyColorMap(feat, cv2.COLORMAP_HSV)
        img = img.numpy()
        img = np.transpose(img, (1,2,0))
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.normalize(img, None, alpha=0, beta = 190, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img = np.uint8(img)
        out = cv2.add(0.6*feat, 0.4*img)
        cv2.imwrite(savename)
    # ===================End=======================
    # =============================================

    # ==========Gradient Check Module==============
    # =============================================
    def gradient_check(self, inputs):
        print('================Gradient check begin=================')
        feats = inputs.to(self.device)
        for module in tqdm(self.model.modules()):
            result = gradcheck(module, feats)
            feats = module(feats)
            if result != True:
                print('gradient check failed in {}'.format(module))
        print('===============Gradient check finished===============')
