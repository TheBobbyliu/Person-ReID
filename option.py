import argparse

class configs():
    def __init__(self, filepath):
        attribute = dict()
        with open(filepath ,'r') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                if line == '\n':
                    continue
                if line[0] == '#':
                    continue
                varname, varvalue = line.split(':')
                varname = varname.strip()
                varvalue = varvalue.strip()
                attribute[varname] = varvalue
        for k, v in attribute.items():
            if v.isdigit():
                v = int(v)
            elif v == 'True':
                v = True
            elif v == 'False':
                v = False
            elif v == 'None':
                v = None
            elif v[0]=='[':
                v = v[1:-1].split(',')
                v = [int(f) for f in v]
            setattr(self, k, v)
        self.lr = float(self.lr)
        if 'Triplet' in self.loss:
            self.margin = float(self.margin)
        if self.optimizer == 'ADAM':
            self.beta1 = float(self.beta1)
            self.beta2 = float(self.beta2)
        elif self.optimizer == 'SGD':
            self.momentum = float(self.momentum)
            self.dampening = float(self.dampening)
        self.epsilon = float(self.epsilon)
        self.weight_decay = float(self.weight_decay)
        self.gamma = float(self.gamma)
        if self.random_erasing == True:
            self.probability = float(self.probability)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./config/config.txt', help='config file')

path = parser.parse_args()
args = configs(path.cfg)
