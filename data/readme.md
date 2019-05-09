1.如何自定义sampler呢？
sampler需要输出的只有图片文件的顺序，index
所以可以根据自己的需求来设定。
在这里最好已经封装好data_source数据源
主要设定为：
```
class Sampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
    	super(Sampler, self).__init__(data_source)

    def __iter__(self):
        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image)
        return iter(imgs)

    def __len__(self):
        return len(self._id2index)*self.batch_image
    #staticmethod

    def _sample(population, k):
        if len(poopulation) < k:
            population *= k
        return random.sample(population, k)
```

2. ```from torchvision.datasets.folder import default_loader```
   可以直接打开图片为Image类

3. ```torch.utils.data.Dataset``` 是专门用来读取图片的
   必须要重写的函数有：
```
   def __init__(self, dataset, transform)
   def __len__(self)
   def __getitem__(self, index)
```
4. ```torch.utils.data.dataset.Dataset``` 可以用于定义自己的数据集，不过其实不需要继承这个类也能够用。
	处理自己的数据集可以以whale为思路来做，列出所有的数据，并且把train, val, test编成[image,label]的list格式

5. Data类
    主要用于数据的集合，生成dataloader之后就能够直接iter了。


简单的写一个
```
from torch.utils.data import Dataset, dataloader
class imageDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
  	  self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        img, label = self.dataset[ind]
        img = PIL.Image.open(img)
        if self.transform is not None:
 	      img = self.transform(img)
 	  return img, label

class dataset():
    def __init__(self, args):
        train_folder = args.train_folder
        train, num_ids, num_imgs = _process_folder(train_folder)
        self.train = train

class data:
    def __init__(self):
        train_transform = torchvision.transforms.Compose([transforms.RandomGrayscale(p=0.3), transforms.ToTensor()])
        dataset = dataset()
        self.trainset = dataset.ImageDataset(dataset.train, transform=train_transform)
        self.train_loader = dataloader.DataLoader(self.trainset, batch_size=64, sampler = RandomIdentitySampler, num_workers = 6)

class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
        xxx
```
6.加入transform
transform类最关键的函数为__call__
它不需要继承，直接写object也行，但是__call__函数是主要用于以下情况的
```
a = transform()
b = a(b)
```
