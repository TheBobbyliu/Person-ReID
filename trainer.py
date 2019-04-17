import os
import torch
import numpy as np
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap, write_csv
from utils.re_ranking import re_ranking
from scipy.io import savemat
import pickle
import cv2
import torch.nn as nn
from tqdm import tqdm
from optimizer import optimizer
from time import time

class Trainer():
    def __init__(self, args, model, loss, loader, ckpt, checker):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.testset
        self.queryset = loader.queryset

        self.ckpt = ckpt
        self.model = model
        self.loss = loss
        self.checker = checker
        self.lr = 0.
        self.optimizer = optimizer.make_optimizer(args, self.model)
        self.scheduler = optimizer.make_scheduler(args, self.optimizer)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.load != None:
            self.ckpt.load(self)
            for _ in range(len(ckpt.log)*args.test_every): self.scheduler.step()
        self.epoch = self.scheduler.last_epoch + 1

    # optim is for flexible convert between freeze training and non-freeze training
    def train(self, optim):
        self.scheduler.step()
        self.loss.step()
        self.epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        self.ckpt.write_log('\n[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(self.epoch, lr))
        self.lr = lr
        self.loss.start_log()
        self.model.train()

        # running data
        for batch, (inputs, labels) in enumerate(self.train_loader):
            t1 = time()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optim.zero_grad()
            outputs = self.model(inputs)
            t2 = time()
            # spent a lot of time
            loss = self.loss(outputs, labels)
            loss.backward()
            t3 = time()
            # spent a lot of time
            optim.step()
            t4 = time()

            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                self.epoch, self.args.epochs,
                batch + 1, len(self.train_loader),
                self.loss.display_loss(batch)), 
            end='' if batch+1 != len(self.train_loader) else '\n')
            t5 = time()
            #print('\none: %.4f two: %.4f three: %.4f four: %.4f'%(t2-t1,t3-t2,t4-t3,t5-t4))
        self.loss.end_log(len(self.train_loader))

    def test(self):
        self.epoch = self.scheduler.last_epoch + 1
        self.ckpt.write_log('\n[INFO] Test:')
        self.ckpt.add_log(torch.zeros(1, 5))

        qf = self.extract_feature(self.query_loader)
        gf = self.extract_feature(self.test_loader)

        # save feature, cam, label, frames
        #pytorch_result = {'query_f':qf, 'query_cam':self.queryset.cameras,
        #                  'query_label':self.queryset.ids, 'query_frames':self.queryset.frames,
        #                  'gallery_f':gf, 'gallery_cam':self.testset.cameras,
        #                  'gallery_label':self.testset.ids, 'gallery_frames':self.testset.frames}
        #savemat('./pytorch_result.mat',pytorch_result)
        #print('saved')

        if self.args.re_rank:
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            dist = cdist(qf, gf)
        r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

        self.ckpt.log[-1, 0] = m_ap
        self.ckpt.log[-1, 1] = r[0]
        self.ckpt.log[-1, 2] = r[2]
        self.ckpt.log[-1, 3] = r[4]
        self.ckpt.log[-1, 4] = r[9]
        best = self.ckpt.log.max(0)
        self.ckpt.write_log(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
            m_ap,
            r[0], r[2], r[4], r[9],
            best[0][0],
            (best[1][0] + 1)*self.args.test_every
            )
        )
        if not self.args.test:
            self.ckpt.save(self, self.epoch, is_best=((best[1][0] + 1)*self.args.test_every == self.epoch))


    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
        return inputs.index_select(3,inv_idx)

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        for (inputs, labels) in tqdm(loader):
            # gradient check
            if self.args.gradient_check != 0:
                feats = inputs[1:2]
                print('feats input size: {}'.format(feats.size()))
                feats = feats.double()
                feats.requires_grad = True
                self.checker.model = self.checker.model.double()
                self.checker.gradient_check(feats)
                self.args.gradient_check = 0
            else:
                self.model.eval()

            inputs1 = self.fliphor(inputs)
            input_img1 = inputs1.to(self.device)
            input_img2 = inputs.to(self.device)
            outputs1 = self.model(input_img1)[0]
            outputs2 = self.model(input_img2)[0]
            f1 = outputs1.data.cpu()
            f2 = outputs2.data.cpu()
            #f1 = outputs1[0].data.cpu()
            #f2 = outputs2[0].data.cpu()
            ff = torch.FloatTensor(inputs.size(0), f1.size(1)).zero_()
            ff = ff + f1 + f2

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
        return features.numpy()

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
