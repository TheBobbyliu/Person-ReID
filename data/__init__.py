from torchvision import transforms
from torch.utils.data import dataloader
from datautils.random_erasing import RandomErasing
from sampler import RandomIdentitySampler
from dataset import *

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
            train_list.append(RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_list)

        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = whale.Whale()
        
        self.trainset = ImageDataset(dataset.train, transform = test_transform)
        self.testset = ImageDataset(dataset.gallery, transform = test_transform)
        self.queryset = ImageDataset(dataset.query, transform = test_transform)

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  batch_size = args.batch,
                                                  sampler=RandomIdentitySampler,
                                                  num_workers = args.nThread,
                                                  drop_last = False,
                                                  pin_memory = True)
        self.test_loader = dataloader.DataLoader(self.testset, 
                                                 batch_size=args.batchtest, 
                                                 num_workers=args.nThread,
                                                 drop_last=False,
                                                 pin_memory = True)
        self.query_loader = dataloader.DataLoader(self.queryset,
                                                  batch_size=args.batchtest, 
                                                  num_workers=args.nThread,
                                                  drop_last = False,
                                                  pin_memory=True)