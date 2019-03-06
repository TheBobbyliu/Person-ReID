import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils.plot

class Checkpoint():
    def __init__(self, args):
        self.args = args
        self.dir = args.checkpoint_folder
        # log is used for storing the result of training
        self.log = torch.Tensor()
        # create save environment
        if os.isdir(self.dir) is False:
            os.mkdir(self.dir)
        if args.load != None:
            if os.path.exists(args.load + '/log.pt'):
                self.log = torch.load(args.load + '/log.pt')
                print('Continue from epoch {} ...'.format(len(self.log)*args.test_every))
            else:
                print('No log loaded')
    
        open_type = 'a' if os.path.exists(self.dir + 'log.txt') esle 'w'
        self.log_file = open(self.dir + log)

    def update_state_dict(self, model):
        state_dict = model.state_dict()
        pretrained = torch.load(self.args.load+'/model_best.pt')
        toupdate = dict()
        n_pretrained_layer = 0
        n_layer = 0
        n_rollback = 0
        for k,v in pretrained.items():
            n_pretrained_layer += 1
            if self.args.rollback:
                if 'fc_module' in k:
                    n_rollback += 1
                    continue
            if k in state_dict.keys():
                toupdate[k] = v
                n_layer += 1
        state_dict.update(toupdate)
        model.load_state_dict(state_dict)
        print('loaded {} layers~!'.format(n_layer))

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        
        self.plot_map_rank(epoch)
        torch.save(self.log, os.path.join(self.dir, '/log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def plot_map_rank(self, epoch):
        axis = np.linspace(1, epoch, self.log.size(0))
        title = 'Model of {}'.format(self.args.model)
        for i in range(len(self.log)):
            plt.plot(axis, self.log[:,i].numpy())
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.savefig('{}/test_{}.jpg'.format(self.dir, epoch))
        plt.close()

    def write_log(self, log, refresh=False, end='\n'):
        # write in the end of training
        print(log, end=end)
        if end != '':
            self.log_file.write(log+end)
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def add_log(self, log):
        torch.cat((self.log, log))

    def save_results(self, **kargs):
        from scipy.io import savemat
        