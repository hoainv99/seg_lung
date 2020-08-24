from tools.config import Cfg
from train import Trainer
from model import ResNetUNet
from torch.optim import  Adam
import torch
config = Cfg.load_config_from_file('config/config.yml')
params = {
         'print_every':10,
         'valid_every':5*10,
         'iters':100000,   
         'n_classses' : 2
         }

config['trainer'].update(params)
model = ResNetUNet(config['trainer']['n_classes'])
optimizer = Adam(model.parameters(), lr=1e-6)
train = Trainer(config,model, optimizer,pretrained = False)
x,y = next(iter(train.train_data_loader))
train.train()
