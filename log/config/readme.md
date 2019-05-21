## Config File configuration helper

worker -- the number of workers for I/O operations

cpu -- True/False to determine use CPU for running or not.

nGPU -- (int)To apply Data Parallel in pytorch
```load/save names of model/optimizer/loss state dict```
load -- (str)Checkpoint file for loading, if you only have model state dict, 			please refer to load_model.

save -- (str)Folder to save model/optimizer/loss state_dict

restart -- True/False to determine whether start from epoch 0 or not.
### model configuration
model -- choose a model in folder ./model
### data configuration
datadir -- data folder

height, width -- input size

num_classes -- number of classes in dataset
### network hyperparameters
lr -- learning rate

batchsize -- train batch size

epochs -- total epochs

test -- test_only

test_every -- test during training

batchtest -- test batch size

freeze -- the number of epochs to freeze base parameters. If you want to resume training from a trained model, please set this parameter to 0. 

### ReID specific parameters
batchid -- person identities sampled every mini-batch
#### for PCB/MGN RPP only
module -- choose from MGN/PCB/RPP, see specific description of module in model/pcbrpp.py or other files
 
feats -- (int)feature dimension of every output feature in part models.

slice_p2 -- (int)number of parts of features in MGN p2 section

slice_p3 -- (int)number of parts of features in MGN p3 section

slices -- (int)number of parts of features in PCB

### loss
loss -- see loss/__init__.py for detailed description, this is in the format of weight1*loss_function_name1 + weight2*loss_function_name2+...

margin -- margin value of triplet loss

### optimizer settings
optimizer : ADAM

#### ADAM&&NADAM parameters
beta1 : 0.9

beta2 : 0.999

amsgrad : True
#### SGD parameters (RMSProp only needs momentum)
momentum : 0.9

dampening : 0

nesterov : True

#### universal parameters
epsilon : 0.00000001

weight_decay : 0.0005

lr_decay : 100

### scheduler settings
stepsize : [120, 180]

gamma : 0.1

#optimizer, loss, model are integrated in one save file, 
#but for saparate save files there should be a solution(see the code below)
load_optimizer: None

load_loss: None

load_loss_log: None

load_map_log: None

### other settings
#rollback -- not implemented, but easy enough, just initialize the weights of the last convolution/batchsize layers to achieve better results.

random_erasing -- set to True/False to enable random erasing transformation.

probability -- prob to ramdom erase

re_rank -- not implemented

gradient_check -- not implemented
