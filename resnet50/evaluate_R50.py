import argparse
import os, sys
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#From ImagenetVid Generator
from torchvision import transforms as trfms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from genINV_R50_Eval_v2 import ImageNetVID
from statistics import mean 
import pickle
from resnet50_class_embedding import resnet50

from EmbedDatabase_v3 import EmbedDatabase

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

'''Statistics

Train Dataset - The factors of 15,806 are: 1, 2, 7, 14, 1129, 2258, 7903, 15806
Val Dataset - The factors of 2,618 are: 1, 2, 7, 11, 14, 17, 22, 34, 77, 119, 154, 187, 238, 374, 1309, 2618
'''

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-en','--experiment_name', type=str, help="Name of the experiment")
#parser.add_argument('-pd','--path_to_disk',default='/disk/shravank/imageNet_ResNet50_savedModel/', type=str, help="Path to disk")

#!Needs update from IQ
best_acc1 = 0
path_to_disk = '/disk/shravank/imageNet_ResNet50_savedModel/main/run/'

def main():
    args = parser.parse_args()

    global path_to_disk
    path_to_disk = path_to_disk + args.experiment_name + '/'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed


    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        #model = models.__dict__[args.arch](pretrained=True)
        model = resnet50(pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = resnet50()


    #model = models.resnet50()
    

    #Sets classes to 30
    num_classes = 30
    
    '''if not model.fc.weight.size()[0] == num_class:
        # Replace last layer
        print(model.fc)
        model.fc = torch.nn.Linear(2048, num_class)
        print(model.fc)
    else:
        pass'''
    '''for param in model.parameters():
        param.requires_grad = False'''
    
    '''num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)'''

    '''model.fc = nn.Sequential(
        nn.Linear(num_ftrs,1024),
        nn.ReLU(),
        #nn.Linear(2048,2048),
       # nn.ReLU(),
        nn.Linear(1024,num_classes)
    )'''

    #print(model)
    

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    '''optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)'''

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    '''traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')'''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = "../data/metadata_imgnet_vid_train_n2.pkl"
    #best val dataset has _new
    val_dataset = "../data/metadata_imgnet_vid_val_n2.pkl"
    root_datasets = '/mnt/data1/shravank/datasets/'

    '''train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))'''

    transform_train = trfms.Compose([
        trfms.RandomHorizontalFlip(),
        trfms.ColorJitter(0.2, 0.2, 0.2),
        trfms.ToTensor(),
        normalize
    ])
    
    transform_val = trfms.Compose([
        trfms.ToTensor(),
        normalize
    ])
    
    #Generators
    gen_train = ImageNetVID(root_datasets,train_dataset,split='train',transform=transform_val)
    gen_val = ImageNetVID(root_datasets,val_dataset,split='val',transform=transform_val)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    '''train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)'''

    train_loader = DataLoader(gen_train,batch_size=256,shuffle=False)


    '''tl = iter(train_loader)
    b = next(tl)
    print(b[0].shape,b[1])

    import sys
    sys.exit(0)'''

    '''val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)'''

    val_loader = DataLoader(gen_val,batch_size=256,shuffle=False)
    #val_loader = DataLoader(gen_val,batch_size=119/154,,num_workers=11)

    writer = SummaryWriter(path_to_disk) 

    #Creating instance of embedding database
    ed = EmbedDatabase(d=2048)

    if args.evaluate:
        

        '''val_loader_mini_display = DataLoader(gen_val,batch_size=12,shuffle=True)
        validate(val_loader_mini_display, model, criterion, args, writer, epoch=None,mini_display=True)

        #time python resnet50_imgnetVID.py -a resnet50 --evaluate --resume /disk/shravank/imageNet_ResNet50_savedModel/e200b14lr0.001_pre0aug1/model_best.pth.tar -en debug
        gen_val = ImageNetGeneratorV3(root_datasets,val_dataset,split='val',transform=transform_val)
        val_loader = DataLoader(gen_val,batch_size=64)
        acc1,avg_metrics_epoch_val = validate(val_loader, model, criterion, args, writer, epoch=None)'''

        #Validation call above needs to be fixed

        #TrainEmbeddings

        train_embeddings(val_loader, model, criterion, args, writer, ed,epoch=1,mini_display=False)

        #Calculating embedding class predictability

        topk1,topk5 = ed.idAccuracy()

        print('TOPK1:   ',topk1)
        print('TOPK5:   ',topk5)


        

        return

    

    
    metrics_train_all_epochs = {
        'batch_time': [],
        'data_time': [],
        'losses': [],
        'top1': [],
        'top5': []
    }

    metrics_val_all_epochs = {
        'batch_time': [],
        'losses': [],
        'top1': [],
        'top5': []
    }

    

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        avg_metrics_epoch = train(train_loader, model, criterion, optimizer, epoch, args, writer)

        '''metrics_train_all_epochs['batch_time'].append(avg_metrics_epoch['batch_time'])
        metrics_train_all_epochs['data_time'].append(avg_metrics_epoch['data_time'])
        metrics_train_all_epochs['losses'].append(avg_metrics_epoch['losses'])
        metrics_train_all_epochs['top1'].append(avg_metrics_epoch['top1'])
        metrics_train_all_epochs['top5'].append(avg_metrics_epoch['top5'])'''

        for key in metrics_train_all_epochs.keys():
            metrics_train_all_epochs[key].append(avg_metrics_epoch[key])



        # evaluate on validation set
        acc1,avg_metrics_epoch_val = validate(val_loader, model, criterion, args, writer, epoch)

        '''metrics_val_all_epochs['batch_time'].append(avg_metrics_epoch_val['batch_time'])
        metrics_val_all_epochs['losses'].append(avg_metrics_epoch_val['losses'])
        metrics_val_all_epochs['top1'].append(avg_metrics_epoch_val['top1'])
        metrics_val_all_epochs['top5'].append(avg_metrics_epoch_val['top5'])'''

        for key in metrics_val_all_epochs.keys():
            metrics_val_all_epochs[key].append(avg_metrics_epoch_val[key])

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

    pickle.dump( metrics_train_all_epochs, open( "metrics_res50/metrics_train_all_epochs.pkl", "wb" ) )
    pickle.dump( metrics_val_all_epochs, open( "metrics_res50/metrics_val_all_epochs.pkl", "wb" ) )

    print('---AVG_TRAIN_METRICS---')
    for metric in metrics_train_all_epochs:
        print(metric,mean(metrics_train_all_epochs[metric]))
    
    arg_top1 = np.argmax(np.array(metrics_train_all_epochs['top1']))
    print('Max top1: ',max(metrics_train_all_epochs['top1']),arg_top1)
    print('Top5(argmax-top1)',metrics_train_all_epochs['top5'][arg_top1])
    print('Max top5: ',max(metrics_train_all_epochs['top5']),np.argmax(np.array(metrics_train_all_epochs['top5'])))

    print('---AVG_VAL_METRICS---')
    for metric in metrics_val_all_epochs:
        print(metric,mean(metrics_val_all_epochs[metric]))

    arg_top1 = np.argmax(np.array(metrics_val_all_epochs['top1']))
    print('Max top1: ',max(metrics_val_all_epochs['top1']),arg_top1)
    print('Top5(argmax-top1)',metrics_val_all_epochs['top5'][arg_top1])
    print('Max top5: ',max(metrics_val_all_epochs['top5']),np.argmax(np.array(metrics_val_all_epochs['top5'])))



def train_embeddings(val_loader, model, criterion, args, writer, ed, epoch=1,mini_display=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    #Validate only to display 12 images

    with torch.no_grad():
        end = time.time()
        for i, (images, target,ids) in enumerate(val_loader):
            images = images.float()
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            

            '''print(images.dtype)
            print(target.dtype)
            print(ids.dtype)
            sys.exit(0)'''

            # compute output
            output, embedding = model(images)
            loss = criterion(output, target)
            
            ed.addIndex(embedding.cpu().numpy(),ids.cpu().numpy())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            #Stops the validation after just one batch required for mini display    
            if mini_display:

                _, pred = output.topk(5, 1, True, True)
                figure = log_images(images,pred,target)
                writer.add_figure('Mini-Display', figure)
                

                if i == 0:
                    break


        if isinstance(top1.avg,int):
            t1 = top1.avg
            t5 = top5.avg
        else:
            t1 = top1.avg.item()
            t5 = top5.avg.item()

        if epoch is not None:
            writer.add_scalar('Validate/Loss', losses.avg, epoch)
            writer.add_scalar('Validate/Accuracy/Top1', t1, epoch)
            writer.add_scalar('Validate/Accuracy/Top5', t5, epoch)
            writer.flush()
        #else:
            #Add writer for validation only run

        avg_metrics_epoch = {
            'batch_time': batch_time.avg,
            'losses': losses.avg,
            'top1': t1,
            'top5': t5
        }

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg,avg_metrics_epoch
       
        
    
    


def validate_embeddings(val_loader, model, criterion, args, writer, epoch, mini_display=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    #Validate only to display 12 images

    with torch.no_grad():
        end = time.time()
        for i, (images, target, ids) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, embedding = model(images)
            loss = criterion(output, target)

            embedDB.addQueryEmbedding(embedding.numpy(),ids)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            #Stops the validation after just one batch required for mini display    
            if mini_display:

                _, pred = output.topk(5, 1, True, True)
                figure = log_images(images,pred,target)
                writer.add_figure('Mini-Display', figure)
                

                if i == 0:
                    break


        if isinstance(top1.avg,int):
            t1 = top1.avg
            t5 = top5.avg
        else:
            t1 = top1.avg.item()
            t5 = top5.avg.item()

        if epoch is not None:
            writer.add_scalar('Validate/Loss', losses.avg, epoch)
            writer.add_scalar('Validate/Accuracy/Top1', t1, epoch)
            writer.add_scalar('Validate/Accuracy/Top5', t5, epoch)
            writer.flush()
        #else:
            #Add writer for validation only run

        avg_metrics_epoch = {
            'batch_time': batch_time.avg,
            'losses': losses.avg,
            'top1': t1,
            'top5': t5
        }

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg,avg_metrics_epoch

def log_images(images,pred,target):

    import matplotlib.pyplot as plt
    import pandas as pd

    map_vid = pd.read_pickle("../data/map_vid.pkl")
    map_vid = map_vid.to_dict()['category_name']

    pred_num = pred.cpu().numpy().astype(int)
    target_num = target.cpu().numpy().astype(int).reshape(-1,1)
    pred_target = np.concatenate((target_num,pred_num),axis=1)

    inv_normalize = trfms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.255]
                    )
    
    images = images.cpu()
    for ind in range(images.size(0)):
        images[ind] = inv_normalize(images[ind])

    figure = plt.figure(figsize=(10,10))
    figure.suptitle('Target Label / Prediction Label')

    for i in range(images.size(0)):
        target = map_vid[pred_target[i,0]]
        prediction = map_vid[pred_target[i,1]]
        title = target+'/'+prediction
        plt.subplot(3,4,i+1,title=title)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i].permute(1,2,0).numpy())
    
    return figure




def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    
    torch.save(state,path_to_disk+filename)
    if is_best:
        shutil.copyfile(path_to_disk+filename,path_to_disk+'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr
    #lr = max(new_lr,0.001)
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
