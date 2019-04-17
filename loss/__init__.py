import os
import numpy as np
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from time import time
from loss.triplet import TripletLoss, TripletSemihardLoss

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckpt):
        super(Loss, self).__init__()
        print('[INFO] Making loss...')

        self.nGPU = args.nGPU
        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'CrossEntropy' or loss_type == 'RPP':
                loss_function = nn.CrossEntropyLoss()
            elif loss_type == 'Triplet':
                loss_function = TripletLoss(args.margin)

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
                })
            

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(self.device)
        if not args.cpu and args.nGPU > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.nGPU)
            )

    def forward(self, outputs, labels):
        losses = []
        for i, l in enumerate(self.loss):
            if l['type'] == 'Triplet':
                loss = l['function'](outputs[0], labels)
                #loss = [l['function'](output, labels) for output in outputs[1:-1]]
                #loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'CrossEntropy':
                #loss = l['function'](outputs[1], labels)
                loss = [l['function'](output, labels) for output in outputs[-1]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'RPP': 
                output_p2 = outputs[-2]
                output_p3 = outputs[-1]
                size_p2 = output_p2.size()
                size_p3 = output_p3.size()
                # flatten to be (batch_size, logits_dim, length)
                output_p2 = output_p2.view(size_p2[0], size_p2[1], -1)
                output_p3 = output_p3.view(size_p3[0], size_p3[1], -1)

                target_p2 = torch.zeros((size_p2[0],size_p2[2],size_p2[3])).type(torch.cuda.LongTensor)
                height_split0 = 0
                for dim in range(0, size_p2[1]):
                    # create target
                    height_split1 = int(size_p2[2]*((dim+1)/size_p2[1]))
                    target_p2[...,height_split0:height_split1,:] = target_p2.new_full(target_p2[...,dim,height_split0:height_split1,:].size(), dim)
                    height_split0 = height_split1

                target_p3 = torch.zeros((size_p3[0],size_p3[2],size_p3[3])).type(torch.cuda.LongTensor)
                height_split0 = 0
                for dim in range(1, size_p3[1]):
                    # create target
                    height_split1 = int(size_p3[2]*((dim+1)/size_p3[1]))
                    target_p3[...,height_split0:height_split1,:] = target_p3.new_full(target_p3[...,dim,height_split0:height_split1,:].size(), dim)
                    height_split0 = height_split1

                # flatten to be (batch_size, logits_dim, length)
                target_p2 = target_p2.view((target_p2.size()[0],-1))
                target_p3 = target_p3.view((target_p3.size()[0],-1))
                loss1 = l['function'](output_p2, target_p2)
                loss2 = l['function'](output_p3, target_p3)
                effective_loss = ((loss1+loss2)*l['weight'])
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            else:
                pass
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, batches):
        self.log[-1].div_(batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.jpg'.format(apath, l['type']))
            plt.close(fig)

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def get_loss_module(self):
        if self.nGPU == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        self.load_state_dict(torch.load(os.path.join(apath, 'loss.pt')))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
