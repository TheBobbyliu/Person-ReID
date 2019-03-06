import argparse

parser = argparse.ArgumentParser(description='Pytorch model')

# basic settings
parser.add_argument('--worker', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--cpu', type='store_true', help='use cpu only')
parser.add_argument('--nGPU', type=int, default=1, help='number of gpus')

# data settings
parser.add_argument('--data_folder', type=str, help='data folder')
parser.add_argument('--data_train', type=str, help='data for training')
parser.add_argument('--data_val', type=str, help='data for validation')
parser.add_argument('--data_test', type=str, help='data for testing')

# network hyperparameters
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='batch size')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--test', type='store_true', help='test only')
parser.add_argument('--test_every', type=int, default = 25, help='test every n epochs')
parser.add_argument('--betchtest', type=int, default=16, help='batchsize for testing')

# for re-id
parser.add_argument('--batchid', type=int, default=8, help='ids in every batch')
parser.add_argument('--batchimage', type=int, default=4, help='image number fed into network for every id')

# optimizer&&loss settings
parser.add_argument('--stepsize', type=int, default=[20,40], nargs='+', help='stepsize to decay learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
parser.add_argument('--gamma', type=float, help='learning rate decay')
parser.add_argument('--loss', type=str, default='1*CrossEntropy', help='weight1*loss1 + weight2*loss2')

# save && load
parser.add_argument('--checkpoint_folder', type=str, default='./checkpoint', help='the folder to save model')
parser.add_argument('--load', type=str, default=None, help='the folder to load model and log')
parser.add_argument('--save', type=int, default=None)
parser.add_argument('--resume', type=int, default=-1)

# fundamental
parser.add_argument('--model', type=str, default='mgn', help='choose model')
parser.add_argument('--loss', type=str, default='')

# other setttings 
parser.add_argument('--rollback', type='store_true', help='reset the fully connected layer')
parser.add_argument('--random_erasing', type='store_true')
args = parser.parse_args()
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] == False