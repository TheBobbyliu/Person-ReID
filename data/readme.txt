1.����Զ���sampler�أ�
sampler��Ҫ�����ֻ��ͼƬ�ļ���˳��index
���Կ��Ը����Լ����������趨��
����������Ѿ���װ��data_source����Դ
��Ҫ�趨Ϊ��
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


2. from torchvision.datasets.folder import default_loader
   ����ֱ�Ӵ�ͼƬΪImage��

3. torch.utils.data.Dataset ��ר��������ȡͼƬ��
   ��Ҫ����Ҫ�У�
   def __init__(self, dataset, transform)
   def __len__(self)
   def __getitem__(self, index)

4. torch.utils.data.dataset.Dataset �������ڶ����Լ������ݼ���������ʵ����Ҫ�̳������Ҳ�ܹ��á�
	�����Լ������ݼ�������whaleΪ˼·�������г����е����ݣ����Ұ�train, val, test���[image,label]��list��ʽ

5. Data��
    ��Ҫ�������ݵļ��ϣ�����dataloader֮����ܹ�ֱ��iter�ˡ�


�򵥵�дһ��
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

6.����transform
transform����ؼ��ĺ���Ϊ__call__
������Ҫ�̳У�ֱ��дobjectҲ�У�����__call__��������Ҫ�������������
a = transform()
b = a(b)
