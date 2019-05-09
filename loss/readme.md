如何自己修改loss函数？
```
Class Loss(nn.modules.loss._Loss):
Def __init__(self, args, ckpt):
    Super(Loss, self).__init__()
Self.loss_module = nn.ModuleList()
Self.loss_module.append(nn.CrossEntropyLoss())
Self.loss_module.append(TripletLoss())
Self.device = torch.device(‘cuda’)
Self.loss_module.to(self.device)
If args.nGPU >1:
Self.loss_module = nn.DataParallel(self.loss_module, range(args.nGPU)
Def forward(self, outputs, labels):
Return loss
```
如何记录loss的变化呢？

创建一个```log = torch.Tensor()```,在每一轮都进行记录，concat上去一个值

自定义loss函数

继承抽象类```nn.Module```
