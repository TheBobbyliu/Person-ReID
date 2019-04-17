from option import args
import model
import loss
import optimizer
import data
import trainer
from utils import checkpoint, check
from optimizer import optimizer

from setproctitle import setproctitle
setproctitle('dilation REID')

def main(args):
    ckpt_ = checkpoint.Checkpoint(args)
    # data loader
    dataloader_ = data.Data(args)
    # model build up
    model_ = model.Model(args, ckpt_)
    # loss setting
    loss_ = loss.Loss(args, ckpt_)
    # check module for visualization and gradient check
    check_ = check.check(model_)
    # class for training and testing
    trainer_ = trainer.Trainer(args, model_, loss_, dataloader_, ckpt_, check_)
    if args.test:
        trainer_.test()
        return

    # train
    # train with freeze first 
    if args.freeze > 0:
        print('freeze base_params for {} epochs'.format(args.freeze))
    for par in ckpt_.base_params:
        par.requires_grad = False
        if hasattr(model_.get_model(), 'base_params'):
            for par in model_.get_model().base_params:
                par.requires_grad = False

    optim_tmp = optimizer.make_optimizer(args, model_)
    for i in range(args.freeze):
        trainer_.train(optim_tmp)

    # start training
    for par in model_.parameters():
        par.requires_grad = True

    for i in range(trainer_.epoch, args.epochs):
        trainer_.train(trainer_.optimizer)
        if args.test_every != 0 and (i+1) % args.test_every == 0:
            trainer_.test()

if __name__ == '__main__':
    main(args)
