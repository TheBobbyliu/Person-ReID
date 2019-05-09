Dataset is the son class of torch.utils.data.dataset.Dataset

是最开始的操作对象，主要用于数据集的分类与预处理。

外部提取信息的方法：
```
__getitem__(self, index):
    return img, target
```
或者
```
dataset = [[path, target], [path,target]]
```
