import torch.optim as optim
from .nadam import Nadam
from .n_adam import NAdam
import torch.optim.lr_scheduler as lrs

def make_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': args.momentum,
            'dampening': args.dampening,
            'nesterov': args.nesterov
            }
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon,
            'amsgrad': 1
        }
    elif args.optimizer == 'NADAM':
        optimizer_function = NAdam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'eps': args.epsilon,
            'momentum': args.momentum
        }
    else:
        raise Exception()

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, optimizer):
    #Sets the learning rate of each parameter group to the initial lr
    #decayed by gamma every step_size epochs. When last_epoch=-1, sets
    #initial lr as lr.
    if isinstance(args.stepsize, list):
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones = args.stepsize,
            gamma = args.gamma)
    else:
        scheduler = lrs.StepLR(
            optimizer,
            step_size=args.stepsize,
            gamma=args.gamma
        )
    # there are many other schedulers also easy to realize, see
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
    return scheduler
