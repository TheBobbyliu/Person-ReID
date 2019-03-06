import os
import torch
import numpy as np
import utils.utility as utility
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap, write_csv
from utils.re_ranking import re_ranking
from scipy.io import savemat
import pickle
import cv2
import torch.nn as nn
from tqdm import tqdm
class Trainer():
    def __init__(self, args, model, loss, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        self.query_loader = loader.query_loader
        self.test_loader = loader.test_loader
        self.trainset = loader.trainset
        self.testset = loader.testset
        self.queryset = loader.queryset
        if self.args.final_test:
            self.finaltest_loader = loader.finaltest_loader
            self.finaltestset = loader.finaltestset

        self.ckpt = ckpt
        self.model = model
        self.loss = loss
        self.lr = 0.
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckpt.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckpt.log)*args.test_every): self.scheduler.step()
        self.epoch = self.scheduler.last_epoch + 1

    def train(self):
        self.scheduler.step()
        self.loss.step()
        self.epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(self.epoch, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()
        # freeze RPP
        if self.args.module == 'RPP':
            if self.epoch<self.args.freeze:
                for module in self.model.modules():
                    for par in module.parameters():
                        par.requires_grad = False
                for module in self.model.model.module.mod.modules():
                    for par in module.parameters():
                        par.requires_grad = True
            else:
                for module in self.model.modules():
                    for par in module.parameters():
                        par.requires_grad = True

        # running data
        for batch, (inputs, edges, labels) in enumerate(self.train_loader):
            inputs = torch.cat((inputs, edges),dim=1)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                self.epoch, self.args.epochs,
                batch + 1, len(self.train_loader),
                self.loss.display_loss(batch)), 
            end='' if batch+1 != len(self.train_loader) else '\n')

        self.loss.end_log(len(self.train_loader))

    def test(self):
        if self.args.output_pic:
            if self.args.module=='RPP':
                with torch.no_grad():
                    self.extract_RPP(self.query_loader)

        self.epoch = self.scheduler.last_epoch + 1
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()
        self.ckpt.add_log(torch.zeros(1, 5))

        if not self.args.final_test:
            # if is new, qnew result will be ==== 0 ====
            qf, qnew = self.extract_feature(self.query_loader)
            gf, gnew = self.extract_feature(self.test_loader)
        else:
            qf, qnew = self.extract_feature(self.finaltest_loader)
            gf, gnew = self.extract_feature(self.train_loader)

        if self.args.re_rank:
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            dist = cdist(qf, gf)

        if not self.args.final_test:
            querylabels = self.queryset.ids
            gallerylabels = self.testset.ids
            label2pid = pickle.load(open('../whale/label2pid.pickle','rb'))
            for i in range(len(querylabels)):
                querylabels[i] = label2pid[querylabels[i]]
            for i in range(len(gallerylabels)):
                gallerylabels[i] = label2pid[gallerylabels[i]]
            #distdict = {'distance':dist, 'qid':querylabels, 'gid':gallerylabels}
            #savemat('distance_matrix.mat',distdict)

        if self.args.final_test:
            write_csv(dist, self.finaltestset.ids, self.trainset.ids, qnew, gnew)
            print('Finished Writing CSV File!')
            return

        r = cmc(dist, self.queryset.ids, self.testset.ids, topk = 100, first_match_break=True)
        m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids)

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
        if not self.args.test_only:
            self.ckpt.save(self, self.epoch, is_best=((best[1][0] + 1)*self.args.test_every == self.epoch))

    # flip horizontally
    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
        return inputs.index_select(3,inv_idx)

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        binary_labels = torch.LongTensor()
        for (inputs, edges, labels) in tqdm(loader):
            ff = torch.FloatTensor(inputs.size(0), int(self.args.feats*(3+self.args.slice_p2+self.args.slice_p3))).zero_()
            bb = None
            inputs = torch.cat((inputs, edges),dim = 1)
            for i in range(2):
                if i==1:
                    inputs = self.fliphor(inputs)
                input_img = inputs.to(self.device)
                outputs = self.model(input_img)
                f = outputs[0].data.cpu()
                ff = ff + f
                #binary
                if i == 0:
                    bb = torch.argmax(outputs[1].data.cpu(),1)

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
            binary_labels = torch.cat((binary_labels,bb), 0)

        return features.numpy(), binary_labels.numpy()

    # get a set of images transformed by STN to see how it works
    def extract_RPP(self, loader):
        features = None
        labels = None
        for ind, (inputs, labels) in enumerate(loader):
            input_img = inputs.to(self.device)
            outputs = self.model.model.module.backone(input_img)
            p3 = self.model.model.module.p3(outputs)
            p3_results = self.model.model.module.mod.conv1x1_p3(p3)
            p3_results = self.model.model.module.mod.softmax(p3_results)
            features = p3_results.cpu().data.numpy()
            imgs = inputs.numpy()
            labels = labels.cpu().data.numpy()
            a = 0
            if not os.path.isdir('./result_{}_{}_{}'.format(self.args.slice_p2, self.args.slice_p3,self.epoch)):
                os.mkdir('./result_{}_{}_{}'.format(self.args.slice_p2, self.args.slice_p3,self.epoch))
            for i in range(imgs.shape[0]):
                feat = features[i]
                img = imgs[i]
                feat = np.argmax(feat,axis=0)
                feat = cv2.normalize(feat, None, alpha=60, beta = 180, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                feat = np.repeat(feat.reshape((feat.shape[0],feat.shape[1],1)),3, axis=2)
                feat = cv2.resize(feat, (384,128))
                feat[:,:,0] += 30
                feat[:,:,1] -= 30
                feat = np.uint8(feat)
                feat = cv2.applyColorMap(feat, cv2.COLORMAP_HSV)
                img = np.transpose(img, (1,2,0))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.normalize(img, None, alpha=0, beta = 190, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                img = np.uint8(img)
                out = cv2.add(0.6*feat, 0.4*img)
                cv2.imwrite('./result_{}_{}_{}/e{}_label{}_{}.jpg'.format(self.args.slice_p2, self.args.slice_p3, self.epoch, ind, labels[i], a), out)
                a += 1
            if ind >= 10:
                break
           

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
