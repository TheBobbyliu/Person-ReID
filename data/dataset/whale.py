from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave
from torch.utils.data import dataset
import pickle
class whale(dataset.Dataset):
    """
    Whale Dataset

    Dataset statistics:
    # identities: 5004 + new_whale
    # images: 25361
    # train images = 25361
    # gallery images = 20356
    # query images = 2932

    """
    dataset_dir = '../'

    def __init__(self, args):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'whale/train')
        self.test_dir = osp.join(self.dataset_dir, 'whale/test')
        self.list_train_path = osp.join(self.dataset_dir, 'whale/train.csv')
        self.list_query_path = osp.join(self.dataset_dir, 'whale/query.csv')
        self.list_gallery_path = osp.join(self.dataset_dir, 'whale/gallery.csv')
        with open(osp.join(self.dataset_dir, 'whale/pid2label.pickle'), 'rb') as handle:
            self.pid2label = pickle.load(handle)
        self._check_before_run()
        test_imgs = glob.glob(self.test_dir + '/*')
        test = []
        for ind, path in enumerate(test_imgs):
            test.append((path, os.path.basename(path)))

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, self.list_train_path)
        query, num_query_pids, num_query_imgs = self._process_dir(self.train_dir, self.list_query_path)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.train_dir, self.list_gallery_path)

        num_total_pids = num_train_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs + len(test_imgs)

        if 1:
            print("=> Whale Dataset loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  test     |       | {:8d}".format(len(test_imgs)))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
        self.test = test

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()[1:]
        dataset = []
        label_container = set()
        
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.replace('\n','').split(',')
            img_path = osp.join(dir_path, img_path)
            label = self.pid2label[pid]
            dataset.append((img_path, label))
            label_container.add(label)
        num_imgs = len(dataset)
        num_pids = len(label_container)
        # check if pid starts from 0 and increments with 1
        return dataset, num_pids, num_imgs
