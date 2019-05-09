from torch.optim import lr_scheduler

class Warmup_scheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma, start_lr, warmup_epoch, last_epoch = -1):
        super(Warmup_scheduler, self).__init__(optimizer, last_epoch)
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch
        self.start_lr = start_lr
    
    def get_lr(self):
        if self.last_epoch <= self.warmup_epoch:
            lrs = [self.start_lr + (base_lr-self.start_lr)*
                   self.last_epoch/self.warmup_epoch 
                   for base_lr in self.base_lrs]
        else:
            lrs = [base_lr*self.gamma**(self.last_epoch//self.step_size) 
                   for base_lr in self.base_lrs]
        return lrs