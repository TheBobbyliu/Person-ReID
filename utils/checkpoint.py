import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import datetime
from .utility import load_state_dict

class Checkpoint():
    def __init__(self, args):
        self.args = args
        self.savedir = args.save
        self.loaddir = args.load
        # log is used for storing the result of training
        self.log = torch.Tensor()
        # base_params is concerned with freeze base training
        self.base_params = []
        
        # load model&log&optimizer
        state = None
        if args.load != None:
            if not os.path.exists(self.loaddir):
                print('Loading model with unaccessible path!')
                return
            else:
                state = torch.load(self.loaddir)
        self.state = state

        # create save environment
        if os.path.isdir(self.savedir) is False:
            os.mkdir(self.savedir)

        open_type = 'a' if os.path.exists(self.savedir + 'log.txt') else 'w'
        self.log_file = open(self.savedir + '/log.txt', open_type)
        with open(self.savedir + '/config.txt', open_type) as f:
            f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def update_model_state_dict(self, model):
        state_dict = model.state_dict()
        if 'model_state_dict' in self.state.keys():
            pretrained = self.state['model_state_dict']
        elif 'state_dict' in self.state.keys():
            pretrained = self.state['state_dict']
        else:
            pretrained = self.state

        self.base_params = load_state_dict(model, pretrained, self.args)

    def save(self, trainer, epoch, is_best=False):
        trainer.loss.plot_loss(self.savedir, epoch)
        self.plot_map_rank(epoch)
        loss_state_dict = trainer.loss.get_loss_module().state_dict()
        loss_log = trainer.loss.log
        model_state_dict = trainer.model.get_model().state_dict()
        save_dict = {
            'loss_state_dict': loss_state_dict,
            'loss_log': loss_log,
            'model_state_dict': model_state_dict,
            'optimizer_log':self.log,
            'optimizer_state_dict':trainer.optimizer.state_dict(),
            'epoch':epoch,
            'is_best':is_best
        }
         
        torch.save(save_dict, os.path.join(self.savedir, 'checkpoint_{}.pt'.format(epoch)))
        if is_best:
            torch.save(save_dict, os.path.join(self.savedir, 'checkpoint_best.pt'))
    
    def load(self, trainer):
        # we load model in 2 ways: 
        # 1. load one file including all state_dict and logs
        # 2. load several files respectively for optimizer, model and loss
        # optimizer logger
        if not self.args.restart:
            if 'optimizer_log' in self.state.keys() and self.args.restart is False:
                self.log = self.state['optimizer_log']
                print('Continue from epoch {} ...'.format(len(self.log)*self.args.test_every))
            else:
                if self.args.load_map_log != None and self.args.restart is False:
                    try:
                        self.log = torch.load(self.args.load_map_log)
                        print('Continue from epoch {} ...'.format(len(self.log)*self.args.test_every))
                    except FileNotFoundError:
                        print('No optimizer log loaded')
                else:
                    print('No optimizer log loaded')
        else:
            print('No optimizer log loaded')
        # optimizer state dict
        if 'optimizer_state_dict' in self.state.keys():
            if self.args.restart:
                print('choose to restart, no optimizer loaded')
            else:
                trainer.optimizer.load_state_dict(self.state['optimizer_state_dict'])
        else:
            if self.args.load_optimizer != None:
                try:
                    trainer.optimizer.load_state_dict(torch.load(self.args.load_optimizer))
                except FileNotFoundError:
                    print('No optimizer log loaded')
            else:
                print('No optimizer log loaded')
        # loss state dict
        if 'loss_state_dict' in self.state.keys():
            trainer.loss.get_loss_module().load_state_dict(self.state['loss_state_dict'])
        else:
            if self.args.load_loss != None:
                try:
                    trainer.loss.get_loss_module().load_state_dict(torch.load(self.args.load_loss))
                except FileNotFoundError:
                    print('No loss loaded')
            else:
                print('No loss loaded')
        # loss logger
        if 'loss_log' in self.state.keys() and self.args.restart is False:
            trainer.loss.log = self.state['loss_log']
        else:
            if self.args.load_loss_log != None and self.args.restart is False:
                try:
                    trainer.loss.log = torch.load(self.args.load_loss_log)
                except FileNotFoundError:
                    print('No loss log loaded')
            else:
                print('No loss log loaded')

        self.update_model_state_dict(trainer.model)

    def plot_map_rank(self, epoch):
        axis = np.linspace(1, epoch, self.log.size(0))
        title = 'Model of {}'.format(self.args.model)
        for i in range(len(self.log)):
            plt.plot(axis, self.log[:,i].numpy())
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.savefig('{}/test_{}.jpg'.format(self.savedir, epoch))
        plt.close()

    def write_log(self, log, refresh=False, end='\n'):
        # write in the end of training
        print(log, end=end)
        if end != '':
            self.log_file.write(log+end)
        if refresh:
            self.log_file.close()
            self.log_file = open(self.savedir + '/log.txt', 'a')

    def add_log(self, log):
        self.log = torch.cat((self.log, log))
