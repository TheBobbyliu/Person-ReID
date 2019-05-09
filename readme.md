# Person Re-Identification Models 
written by Pytorch

### This is a conclusive repository recording my way of learning pytorch.
#### Hope it is also helpful for you.

requirement:
```
  Pytorch version: 1.0.0
  Python version: 3.5.0 and above
```

Person re-identification is an engineering application of deep neural networks on problems like person
 search, Multi-person tracking.
 
For training and testing models, we can configure ```./config/config.txt``` , some parameters are 
introduced as follows:
```
model : select a model in path ./model/
loss : input the losses and their weights
```

It might be upsetting that for training, we have to manually edit ```./loss/__init__.py''' (line 58-60, 65-67)
 so as to adjust to different outputs, although this is very easy.
 
For training and testing , after setting up ```config.txt```, write:
```
  python3 main.py --cfg config/config.txt
```



Test Results of some models on Market-1501 dataset are listed as the following:

|MODEL|mAP|rank-1|rank-3|rank-5|rank-10|
|------|------|------|------|------|------|
|AMG_front|0.5887|0.7936|0.8884|0.9178|0.9471|
|MGN with p2=3&p3=4|0.8421|0.9365|0.9688|0.9771|0.9857|
|ResNet50|0.6647|0.8438|0.9138|0.9403|0.9611|
|ResNet50-mid|0.7027|0.8628|0.9267|0.9486|0.9682|
|PCB|0.8128|0.9305|0.9623|0.9724|0.9849|
|PCB with rollback|0.8232|0.9362|0.9659|0.9742|0.984|

PS. In this repository, MGN's base model is MobileNetV2, but not the ResNet50.

Rollback is a trick of training by:
```
Youngmin Ro, Jongwon Choi, Dae Ung Jo, Byeongho Heo, Jongin Lim, Jin Young Choi, " Backbone Can Not be Trained at Once: Rolling Back to Pre-trained Network for Person Re-Identification", CoRR, 2019. (AAAI at 2019 Feb.)
```
