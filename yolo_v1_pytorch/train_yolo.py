import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from nn_view import View

from voc import VOCDataset
from darknet import DarkNet
from yolo_v1 import YOLOv1
from loss import Loss
import sys

import os,sys,socket
import numpy as np
import math
from datetime import datetime

from r50_yolo import resnet50

#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
sys.path.append('..')
from genINV_Yolo_v2 import ImageNetVID

from pytorch_model_summary import summary

torch.autograd.set_detect_anomaly(True)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Check if GPU devices are available.
use_gpu = torch.cuda.is_available()
assert use_gpu, 'Current implementation does not support CPU mode. Enable CUDA.'
print('CUDA current_device: {}'.format(torch.cuda.current_device()))
print('CUDA device_count: {}'.format(torch.cuda.device_count()))

# Path to data dir.
#image_dir = 'data/VOC_allimgs/'

# Path to label files.
#train_label = ('data/voc2007.txt', 'data/voc2012.txt')
#val_label = 'data/voc2007test.txt'

# Path to checkpoint file containing pre-trained DarkNet weight.
#checkpoint_path = 'weights/darknet/model_best.pth.tar'

path_to_disk = '/mnt/data1/shravank/results/locem/main/run/'
experiment_name = 'g5_yolob_e300b64_v1.03'
#experiment_name = 'g5_yolob_e300b192_v1'              #'g4_yolob_e300b64_v1'
path_to_disk = path_to_disk + experiment_name + '/'
# Frequency to print/log the results.
print_freq = 5
tb_log_freq = 5

#Network Parameteres
S=7
B=2
X=5
C=30
image_size = 448

# Training hyper parameters.
init_lr = 0.001
base_lr = 0.001
momentum = 0.9
weight_decay = 5e-4
num_epochs = 300
batch_size = 64

'''def class_decoder(self,pred_tensor,target_tensor,accuracy):

   
    S,B,C,X,gamma = self.S,self.B,self.C,self.X,self.gamma

    #Extract the mask of targets with a gamma value 1,2,3. This will give us a location as to where the boxes are and their class embedding respectively
    #gamma should be at 40?
    #For all triplets only
    #class_mask_target = (target_tensor[:,:,:,B*X+C]==1) | (target_tensor[:,:,:,B*X+C]==2) | (target_tensor[:,:,:,B*X+C]==3)
    #For all objects
    class_mask_target = (target_tensor[:,:,:,4]==1) | (target_tensor[:,:,:,9]==1)

    #class_mask_target tensor is used to identify the gamma boxes in pred_tensor
    pred_tensor_gamma = pred_tensor[class_mask_target]
    pred_tensor_gamma = pred_tensor_gamma[:,B*X:B*X+C] #We only want the class embeddings [n_objects,C]

    #class_mask_target tensor is used to identify the gamma boxes in target_tensor
    target_tensor_gamma = target_tensor[class_mask_target]
    target_tensor_gamma = target_tensor_gamma[:,B*X:B*X+C] #We only want the class embeddings [n_objects,C]

    #Finds the class label
    target = torch.argmax(target_tensor_gamma, dim=1) #[n_objects,1]
    output = pred_tensor_gamma #[n_objects,C]

    acc1, acc5 = accuracy(output, target, topk=(1, 5))

    return acc1, acc5'''

# Learning rate scheduling.
def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
    elif epoch == 1:
        lr = 0.001
    elif epoch == 15:
        lr = 0.001
    elif epoch == 50: #75
        lr = 0.0001 #0.001
    elif epoch == 80: #105
        lr = 0.00001 #0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Load pre-trained darknet.
'''darknet = DarkNet(conv_only=True, bn=True, init_weight=True)
darknet.features = torch.nn.DataParallel(darknet.features)

src_state_dict = torch.load(checkpoint_path)['state_dict']
dst_state_dict = darknet.state_dict()

for k in dst_state_dict.keys():
    print('Loading weight of', k)
    dst_state_dict[k] = src_state_dict[k]
darknet.load_state_dict(dst_state_dict)

# Load YOLO model.
yolo = YOLOv1(darknet.features)
yolo.conv_layers = torch.nn.DataParallel(yolo.conv_layers)
yolo.cuda()'''

pretrained = True
if pretrained:
    print("=> using pre-trained model '{}'".format('resnet50'))
    #yolo = models.__dict__['resnet50'](pretrained=True)
    yolo = resnet50(pretrained=True,S=S,B=B,C=C,X=X)
else:
    print("=> creating model '{}'".format('resnet50'))
    #yolo = models.__dict__['resnet50']()
    yolo = resnet50(S=S,B=B,C=C,X=X)

#print(yolo)



# show input shape
#print(summary(yolo, torch.rand((10, 3, 448, 448)), show_input=True))

# show output shape
#print(summary(yolo, torch.rand((10, 3, 448, 448)), show_input=False))

# show output shape and hierarchical view of net
#print(summary(yolo, torch.rand((10, 3, 448, 448)), show_input=False, show_hierarchical=True))


#Enable Parallel
yolo = torch.nn.DataParallel(yolo).cuda()
#yolo.to(torch.device('cuda:1'))

# Setup loss and optimizer.
criterion = Loss(feature_size=S, num_bboxes=B, num_classes=C, lambda_coord=5.0, lambda_noobj=0.5)
optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)


train_dataset = "../data/metadata_imgnet_vid_train_n2.pkl"
#best val dataset has _new
val_dataset = "../data/metadata_imgnet_vid_val_n2.pkl"

if socket.gethostname() == 'finity':
    root_datasets = '/mnt/data1/shravank/datasets/'
elif socket.gethostname() == 'iq.cs.uoregon.edu':
    root_datasets = '/disk/shravank/datasets'
else
    raise ValueError('Unknown host')

# Load Pascal-VOC dataset.
gen_train = ImageNetVID(root_datasets,train_dataset,split='train',image_size=image_size,S=S,B=B,C=C,X=X)
train_loader = DataLoader(gen_train,batch_size=batch_size,num_workers=2,shuffle=True)

gen_val = ImageNetVID(root_datasets,val_dataset,split='val',image_size=image_size,S=S,B=B,C=C,X=X)
val_loader = DataLoader(gen_val,batch_size=batch_size)

#print('Number of training images: ', len(train_dataset))

# Open TensorBoardX summary writer
#log_dir = datetime.now().strftime('%b%d_%H-%M-%S')
#log_dir = os.path.join(path_to_disk, log_dir)
log_dir = path_to_disk
writer = SummaryWriter(log_dir=path_to_disk)

# Training loop.
logfile = open(os.path.join(log_dir, 'log.txt'), 'w')
best_val_loss = np.inf

f = open(os.path.join(log_dir,'best_loss_epoch.txt'),'a')

for epoch in range(num_epochs):
    print('\n')
    print('Starting epoch {} / {}'.format(epoch, num_epochs))

    # Training.
    yolo.train()
    total_loss = 0.0
    total_batch = 0

    for i, (imgs, targets) in enumerate(train_loader):
        
        # Update learning rate.
        update_lr(optimizer, epoch, float(i) / float(len(train_loader) - 1))
        lr = get_lr(optimizer)

        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        targets = Variable(targets)
        imgs, targets = imgs.cuda(), targets.cuda()
        #imgs,targets = imgs.to(torch.device('cuda:1')), targets.to(torch.device('cuda:1'))

        #print('imgs size'+str(imgs.size()))
        #print(imgs)

        # Forward to compute loss.
        preds, _ = yolo(imgs)
        #print(preds)      
        loss, loss_class, loss_box = criterion(preds, targets)
        loss_this_iter = loss.item()
        total_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter

        # Backward to update model weight.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print current loss.
        if i % print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
            % (epoch, num_epochs, i, len(train_loader), lr, loss_this_iter, total_loss / float(total_batch)))

        #Custom Tensorboard Logging
        writer.add_scalar('Train/Loss',loss,epoch)
        writer.add_scalar('Train/Class_Loss',loss_class,epoch)
        writer.add_scalar('Train/Boxes_Loss',loss_box,epoch)
        writer.add_scalar('Learning Rate', lr, epoch)

        # TensorBoard.
        '''n_iter = epoch * len(train_loader) + i
        if n_iter % tb_log_freq == 0:
            writer.add_scalar('Train/Loss', loss_this_iter, n_iter)
            #writer.add_scalar('Learning Rate', lr, n_iter)'''

    # Validation.
    yolo.eval()
    val_loss = 0.0
    total_batch = 0

    for i, (imgs, targets) in enumerate(val_loader):
        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        targets = Variable(targets)
        imgs, targets = imgs.cuda(), targets.cuda()
        #imgs,targets = imgs.to(torch.device('cuda:1')), targets.to(torch.device('cuda:1'))

        # Forward to compute validation loss.
        with torch.no_grad():
            preds, _ = yolo(imgs)
        loss, loss_class, loss_box = criterion(preds, targets)
        loss_this_iter = loss.item()
        val_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter
    val_loss /= float(total_batch)

    # Save results.
    logfile.writelines(str(epoch + 1) + '\t' + str(val_loss) + '\n')
    logfile.flush()

    if epoch<102 and epoch%10==0:
        torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_ep'+str(epoch)+'.pth'))

    torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_latest.pth'))
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        
        f.write('Epoch: '+str(epoch)+' '+'Loss: '+str(best_val_loss)+'\n')
        torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_best.pth'))

    # Print.
    print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f'
    % (epoch + 1, num_epochs, val_loss, best_val_loss))

    #Custom Tensorboard Logging
    writer.add_scalar('Validate/Loss',loss,epoch)
    writer.add_scalar('Validate/Class_Loss',loss_class,epoch)
    writer.add_scalar('Validate/Boxes_Loss',loss_box,epoch)

    ''' # TensorBoard.
    writer.add_scalar('test/loss', val_loss, epoch + 1)'''

f.close()
writer.flush()
writer.close()
logfile.close()
