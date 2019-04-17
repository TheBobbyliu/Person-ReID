from torchvision import transforms
from torch.utils.data import dataloader
from .datautils.random_erasing import RandomErasing
from .samplers import RandomSampler
from .dataset import whale, market1501

class Data:
    def __init__(self, args):
        train_transform_list = [
            transforms.RandomGrayscale(p=0.3),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        if args.random_erasing:
            train_transform_list.append(RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_transform_list)

        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # for market1501 dataset
        if not args.test:
            self.trainset = market1501.Market1501(args, train_transform, 'train')
            self.train_loader = dataloader.DataLoader(self.trainset,
                                                  batch_size = args.batchsize,
                                                  sampler=RandomSampler(self.trainset, args.batchid, args.batchsize//args.batchid),
                                                  num_workers = args.worker,
                                                  drop_last = False,
                                                  pin_memory = True)
        else:
            self.trainset = None
            self.train_loader = None

        self.testset = market1501.Market1501(args, test_transform, 'test')
        self.queryset = market1501.Market1501(args, test_transform, 'query')
        
        self.test_loader = dataloader.DataLoader(self.testset, 
                                                 batch_size=args.batchtest, 
                                                 num_workers=args.worker,
                                                 drop_last=False,
                                                 pin_memory = True)
        self.query_loader = dataloader.DataLoader(self.queryset,
                                                  batch_size=args.batchtest, 
                                                  num_workers=args.worker,
                                                  drop_last = False,
                                                  pin_memory=True)
