import argparse
import os, sys
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from r50_locem import resnet18
from r50_locem import resnet50
from r50_locem import resnet101
torch.autograd.set_detect_anomaly(True)

from pytorch_model_summary import summary

S=7
B=2
X=5
C=30
beta=64
gamma=1
image_size = 448

model = resnet50(pretrained=True,S=S,B=B,C=C,X=X,beta=beta)
#model = torch.nn.DataParallel(model).cuda()
model.to(torch.device('cuda:1'))
#imgs.to(torch.device('cuda:1'))
#model.eval()
print(model)

inp = torch.rand(32,3,image_size,image_size)
inp = inp.to(torch.device('cuda:1'))
out = model(inp)

writer = SummaryWriter()
writer.add_graph(model,input_to_model=inp)

# show input shape
print(summary(model,inp, show_input=True))

# show output shape
print(summary(model,inp, show_input=False))

print(out.size())
