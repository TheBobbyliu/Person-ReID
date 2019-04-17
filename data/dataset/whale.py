import os
import glob
import os.path as osp
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
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

    def __init__(self, args, transform, dtype):
        self.transform = transform
        self.loader = default_loader

        data_path = args.datadir
        if dtype == 'train':
            data_path += '/train'
        elif dtype == 'test':
            data_path += '/test'
        elif dtype == 'gallery':
            data_path += '/gallery'
        else:
            data_path += './query'
        # testset in whale identification dataset is saparate and we do not know the actual id of them
        self._check_before_run(data_path)
        self.images = [path for path in list_pictures(data_path)]
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

        if 1:
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  {}    | {:5d} | {:8d}".format(dtype, len(self.unique_ids), len(imgs)))

    def __getitem__(self, index):
        pid = self.id(self.images[index])
        img = self.loader(self.images[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self._id2label[pid]

    def __len__(self):
        return len(self.images)
        
    def _check_before_run(self, path):
        """Check if all files are available before going deeper"""
        if not osp.exists(path):
            raise RuntimeError("'{}' is not available".format(path))

    @staticmethod
    def id(path):
        return int(path.split('/')[-1].split('.')[0])
    
    @property
    def ids(self):
        return [self.id(path) for path in self.images]

    @property
    def unique_ids(self):
        return sorted(set(self.ids))