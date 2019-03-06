import torch
import torch.nn as nn
import numpy as np
from option import args
import model
import loss
import optimizer
import data
import train
from utils import checkpoint

def main(args):
    # continue from previous training
    ckpt = checkpoint.Checkpoint(args)
    # data loader
    loader = data.Data(args)
    # model build up
    model = model.Model(args, ckpt)
    # loss setting
    loss = loss.Loss(args, ckpt) if not args.test else None
    # optimizer
    opt = optimizer.Optimizer(args)
    # class for training and testing
    trainer = trainer.Trainer(args, model, loss, loader, ckpt)

    for i in range(ckpt.current_epoch, args.epochs):
        trainer.train()
        if args.test_every != 0 and (i+1) % args.test_every == 0:
            trainer.test()

if __name__ == '__main__':
    main(args)