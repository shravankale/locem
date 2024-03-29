import argparse
import os, sys
import random
import shutil
import time
import warnings
import numpy as np
import socket
import pandas as pd

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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
torch.autograd.set_detect_anomaly(True)
from torchvision.utils import make_grid
from r50_locem_g6 import resnet50
#from GeM_Pooling_Test.r50_locem import resnet50
from random import random
from evaluate_map import evaluate_retrieval

#For LocEm
from loss import locemLoss
from nn_view import View

#From ImagenetVid Generator
from torchvision import transforms as trfms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
sys.path.append('..')
#from genINV_R50_v1 import ImageNetVID
#from genINV_Locem_v1 import ImageNetVID

#For Eval
from collections import defaultdict
from genINV_Locem_Eval_v2 import ImageNetVID
from detect_locem import locEmDetector
from util.EmbedDatabase_v3 import EmbedDatabase

from statistics import mean 
import pickle,cv2

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


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
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default= 1e-4, type=float,
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
parser.add_argument('-en','--experiment_path', type=str, help="Name of the experiment that contains the best model")
parser.add_argument('--gem', dest='gem', action='store_true',
                    help='add gem pooling layer')
#parser.add_argument('-pd','--path_to_disk',default='/disk/shravank/imageNet_ResNet50_savedModel/', type=str, help="Path to disk")


best_acc1 = 0
path_to_disk = '/mnt/data1/shravank/results/locem/main/run/'

S=7
B=2
X=5
C=30
beta=64
gamma=1
image_size = 448
#gem = False

def rescaleBoundingBox(height,width,rescaled_dim,xmin,ymin,xmax,ymax):
    
    #Required CNN input dimensions are generally squares hence just one dimension, rescaled_dim
    scale_x = rescaled_dim/width
    scale_y = rescaled_dim/height

    xmax = int(xmax * scale_x)
    xmin = int(xmin * scale_x)
    ymax = int(ymax * scale_y)
    ymin = int(ymin * scale_y)
    
    return [xmin,ymin,xmax,ymax]

def collate_fn(data):

    '''print("TYPE DATA COLLATE",type(data))
    print("LEN DATA COLLATE",len(data))
    print("type data[0]",type(data[0][0]))
    print("type data[1]",type(data[0][1]))
    print("type data[2]",type(data[0][2]))'''

    #sys.exit(0)
    '''images = torch.tensor(np.transpose(data[0][0],(2,0,1)))
    bboxes = torch.tensor(data[0][1])'''

    n = len(data[0])
    out = []

    for i in range(n):
        out.append(data[0][i])

    return out

def main():
    args = parser.parse_args()

    global path_to_disk
    path_to_disk = path_to_disk + args.experiment_path + '/'

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

    gem = args.gem

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
        model = resnet50(pretrained=True,S=S,B=B,C=C,X=X,beta=beta,gem=False)
    else:
        print("=> creating model '{}'".format(args.arch))
        #model = models.__dict__[args.arch]()
        model = resnet50(S=S,B=B,C=C,X=X,beta=beta,gem=gem)

    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, final_layer_units)

    #SIGMOID WAS ADDED BECAUSE SOME OF THE PREDICTED VALUES WERE NEGATIVE
    #DURING WH LOSS CALCULATION WHICH TAKES SQUARE ROOT OF WH, NANs WERE INTRODUCED INTO THE LOSS
    #BUT SOME PREDICTED VALUES MIGHT NEED TO BE NEGATIVE
    #https://www.reddit.com/r/deeplearning/comments/9z50qi/confused_about_yolo_loss_function/

    print(model)

    #Importing Embedder Database
    ed = EmbedDatabase()

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
            print('HERE')
            model = torch.nn.DataParallel(model).cuda()
            #model.to(torch.device('cuda:0'))

    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = locemLoss(feature_size=S, num_bboxes=B, num_classes=C, lambda_coord=5.0, lambda_noobj=0.5,beta=beta,gamma=gamma)

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

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = "../data/metadata_imgnet_vid_train_n2.pkl"
    #best val dataset has _new
    val_dataset = "../data/metadata_imgnet_vid_val_n2.pkl"
    
    if socket.gethostname() == 'finity':
        root_datasets = '/mnt/data1/shravank/datasets/'
    elif socket.gethostname() == 'iq.cs.uoregon.edu':
        root_datasets = '/disk/shravank/datasets'
    else:
        raise ValueError('Unknown host')


    '''transform = trfms.Compose([
        #add random crop
        #trfms.RandomHorizontalFlip(),
        #trfms.ColorJitter(0.2, 0.2, 0.2),
        trfms.ToTensor(),
        normalize
    ])'''
    
    #Generators
    gen_train = ImageNetVID(root_datasets,train_dataset,split='train',image_size=image_size,S=S,B=B,C=C,X=X)
    gen_val = ImageNetVID(root_datasets,val_dataset,split='val',image_size=image_size,S=S,B=B,C=C,X=X)


    train_loader = DataLoader(gen_train,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)
    val_loader = DataLoader(gen_val,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)

    print('Len of loader',len(val_loader))
    

    #detector = locEmDetector(args.experiment_path,conf_thresh=0.1, prob_thresh=0.1, nms_thresh=0.30)

    writer = SummaryWriter(path_to_disk)
    ed = EmbedDatabase(d=beta)

    detector = locEmDetector(model,conf_thresh=0.1, prob_thresh=0.1, nms_thresh=0.5,S=S,B=B,C=C,X=X,beta=beta,image_size=image_size)

    loader_mode = 'train_loader'
    #uids_list = []
    
    if loader_mode =='val_loader':

        uids_list = gen_val.uids_list
        aps = new_validate(val_loader, detector, ed,writer,uids_list)
    else:
        uids_list = gen_train.uids_list
        aps = new_validate(train_loader, detector, ed,writer,uids_list)

    topk1,topk5 = ed.idAccuracy()

    print('TOPK1:   ',topk1)
    print('TOPK5:   ',topk5)

    '''print('Mean APS',np.mean(aps))

    map_vid = pd.read_pickle("../data/map_vid.pkl")
    map_cat = map_vid.to_dict()['category_name']
    #dval = pd.read_pickle("../data/metadata_imgnet_vid_val_n2.pkl")
    class_dict = {map_cat[i] for i in map_cat}
    
    class_aps_dict = {}
    for i,j in zip(class_dict,aps):
        class_aps_dict[i]=j
    print(class_aps_dict)'''

    writer.close()

    return

'''def class_decoder(pred_tensor,target_tensor,accuracy):

        if len(pred_tensor)==0:
            return 0,0
        #S,B,C,X = self.S,self.B,self.C,self.X

        #Extract the mask of targets with a gamma value 1,2,3. This will give us a location as to where the boxes are and their class embedding respectively
        #gamma should be at 40?
        #class_mask_target = (target_tensor[:,:,:,B*X+C]==1) | (target_tensor[:,:,:,B*X+C]==2) | (target_tensor[:,:,:,B*X+C]==3)
        class_mask_target = (target_tensor[:,:,4]==1) | (target_tensor[:,:,9]==1)

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

        return acc1, acc5

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
        return res'''

def compute_average_precision(recall, precision):
    """ Compute AP for one class.
    Args:
        recall: (numpy array) recall values of precision-recall curve.
        precision: (numpy array) precision values of precision-recall curve.
    Returns:
        (float) average precision (AP) for the class.
    """
    # AP (AUC of precision-recall curve) computation using all points interpolation.
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i -1], precision[i])

    ap = 0.0 # average precision (AUC of the precision-recall curve).
    for i in range(precision.size - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return ap
       
def evaluate(preds,targets,class_names,threshold=0.5): #original threshold 0.5
    
    """ Compute mAP metric.
    Args:
        preds: (dict) {class_name_1: [[filename, prob, x1, y1, x2, y2], ...], class_name_2: [[], ...], ...}.
        targets: (dict) {(filename, class_name): [[x1, y1, x2, y2], ...], ...}.
        class_names: (list) list of class names.
        threshold: (float) threshold for IoU to separate TP from FP.
    Returns:
        (list of float) list of average precision (AP) for each class.
    my args:
        targets_ev[(filename,classname[b])].append([x1,y1,x2,y2])
        preds_ev[classname_p].append([filename, prob, x1, y1, x2, y2])

    """    
    
    aps = [] # list of average precisions (APs) for each class.
    #aps_dict = defaultdict()

    for class_name in class_names:
        class_preds = preds[class_name] # all predicted objects for this class.

        if len(class_preds) == 0:
            ap = 0.0 # if no box detected, assigne 0 for AP of this class.
            print('---class {} AP {}---'.format(class_name, ap))
            aps.append(ap)
            #aps_dict[class_name]=ap
            continue #CHECK! It should be continue as even if ap is 0 for one class, other classes would have an ap

        image_fnames = [pred[0]  for pred in class_preds]
        probs        = [pred[1]  for pred in class_preds]
        boxes        = [pred[2:] for pred in class_preds]

        sorted_idxs = np.argsort(probs)[::-1]
        image_fnames = [image_fnames[i] for i in sorted_idxs]
        boxes        = [boxes[i]        for i in sorted_idxs]

        # Compute total number of ground-truth boxes for GIVEN CLASS. This is used to compute precision later.
        num_gt_boxes = 0
        for (filename_gt, class_name_gt) in targets:
            if class_name_gt == class_name:
                num_gt_boxes += len(targets[filename_gt, class_name_gt])

        # Go through sorted lists, classifying each detection into TP or FP.
        num_detections = len(boxes)
        tp = np.zeros(num_detections) # if detection `i` is TP, tp[i] = 1. Otherwise, tp[i] = 0.
        fp = np.ones(num_detections)  # if detection `i` is FP, fp[i] = 1. Otherwise, fp[i] = 0.

        for det_idx, (filename, box) in enumerate(zip(image_fnames, boxes)):

            if (filename, class_name) in targets:
                boxes_gt = targets[(filename, class_name)]
                for box_gt in boxes_gt:
                    # Compute IoU b/w/ predicted and groud-truth boxes.
                    inter_x1 = max(box_gt[0], box[0])
                    inter_y1 = max(box_gt[1], box[1])
                    inter_x2 = min(box_gt[2], box[2])
                    inter_y2 = min(box_gt[3], box[3])
                    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
                    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
                    inter = inter_w * inter_h

                    area_det = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
                    area_gt = (box_gt[2] - box_gt[0] + 1.0) * (box_gt[3] - box_gt[1] + 1.0)
                    union = area_det + area_gt - inter

                    iou = inter / union
                    if iou >= threshold:
                        tp[det_idx] = 1.0
                        fp[det_idx] = 0.0

                        boxes_gt.remove(box_gt) # each ground-truth box can be assigned for only one detected box.
                        if len(boxes_gt) == 0:
                            del targets[(filename, class_name)] # remove empty element from the dictionary.

                        break

            else:
                pass # this detection is FP.
        
        # Compute AP from `tp` and `fp`.
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        #tp_cumsum = tp
        #fp_cumsum = fp

        eps = np.finfo(np.float64).eps
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        recall = tp_cumsum / float(num_gt_boxes)

        ap = compute_average_precision(recall, precision)
        print('---class {} AP {}---'.format(class_name, ap))
        aps.append(ap)
        #aps_dict[class_name]=ap

    # Compute mAP by averaging APs for all classes.
    print('---mAP {}---'.format(np.mean(aps)))
    return aps
    #return aps_dict

def evaluate_uid(preds,targets,class_names,threshold=0.5,t=1): #original threshold 0.5
    
    """ Compute mAP metric.
    Args:
        preds: (dict) {class_name_1: [[filename, prob, x1, y1, x2, y2], ...], class_name_2: [[], ...], ...}.
        targets: (dict) {(filename, class_name): [[x1, y1, x2, y2], ...], ...}.
        class_names: (list) list of class names.
        threshold: (float) threshold for IoU to separate TP from FP.
    Returns:
        (list of float) list of average precision (AP) for each class.
    my args:
        targets_ev[(filename,classname[b])].append([x1,y1,x2,y2])
        preds_ev[classname_p].append([filename, prob, x1, y1, x2, y2])

    """    
    
    aps = [] # list of average precisions (APs) for each class.
    #aps_dict = defaultdict()

    for class_name in class_names:
        class_preds = preds[class_name] # all predicted objects for this class.

        if len(class_preds) == 0:
            ap = 0.0 # if no box detected, assigne 0 for AP of this class.
            print('---class {} AP {}---'.format(class_name, ap))
            aps.append(ap)
            #aps_dict[class_name]=ap
            continue #CHECK! It should be continue as even if ap is 0 for one class, other classes would have an ap

        image_fnames = [pred[0]  for pred in class_preds]
        probs        = [pred[1]  for pred in class_preds]
        boxes        = [pred[2:6] for pred in class_preds]
        muids        = [pred[6] for pred in class_preds]

        sorted_idxs = np.argsort(probs)[::-1]
        image_fnames = [image_fnames[i] for i in sorted_idxs]
        boxes        = [boxes[i]        for i in sorted_idxs]
        muids       = [muids[i] for i in sorted_idxs]

        # Compute total number of ground-truth boxes for GIVEN CLASS. This is used to compute precision later.
        num_gt_boxes = 0
        for (filename_gt, class_name_gt) in targets:
            if class_name_gt == class_name:
                num_gt_boxes += len(targets[filename_gt, class_name_gt])

        # Go through sorted lists, classifying each detection into TP or FP.
        num_detections = len(boxes)
        tp = np.zeros(num_detections) # if detection `i` is TP, tp[i] = 1. Otherwise, tp[i] = 0.
        fp = np.ones(num_detections)  # if detection `i` is FP, fp[i] = 1. Otherwise, fp[i] = 0.

        for det_idx, (filename, box, muid) in enumerate(zip(image_fnames, boxes, muids)):

            if (filename, class_name) in targets:
                boxes_gt = targets[(filename, class_name)]
                for box_gt in boxes_gt:
                    # Compute IoU b/w/ predicted and groud-truth boxes.
                    uid_t = box_gt[4]
                    inter_x1 = max(box_gt[0], box[0])
                    inter_y1 = max(box_gt[1], box[1])
                    inter_x2 = min(box_gt[2], box[2])
                    inter_y2 = min(box_gt[3], box[3])
                    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
                    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
                    inter = inter_w * inter_h

                    area_det = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
                    area_gt = (box_gt[2] - box_gt[0] + 1.0) * (box_gt[3] - box_gt[1] + 1.0)
                    union = area_det + area_gt - inter

                    iou = inter / union
                    if (iou >= threshold) and (muid==uid_t):
                        tp[det_idx] = 1.0
                        fp[det_idx] = 0.0

                        boxes_gt.remove(box_gt) # each ground-truth box can be assigned for only one detected box.
                        if len(boxes_gt) == 0:
                            del targets[(filename, class_name)] # remove empty element from the dictionary.

                        break

            else:
                pass # this detection is FP.
        
        # Compute AP from `tp` and `fp`.
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        eps = np.finfo(np.float64).eps
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        recall = tp_cumsum / float(num_gt_boxes)

        ap = compute_average_precision(recall, precision)
        print('---class {} AP {}---'.format(class_name, ap))
        aps.append(ap)
        #aps_dict[class_name]=ap

    # Compute mAP by averaging APs for all classes.
    print('---mAP {}---'.format(np.mean(aps)))
    return aps
    #return aps_dict

def visualize(image,target_boxes,predicted_boxes,writer,n,filename):

    red = (255,0,0)
    green = (0,255,0)
    thickness = 2

    for box in target_boxes:
        x1,y1,x2,y2 = box
        pt1 = (x1,y1)
        pt2 = (x2,y2)
        image = cv2.rectangle(image,pt1,pt2,red,thickness)
    
    if len(predicted_boxes)>0:
        for box in predicted_boxes:
            x1y1, x2y2 = box
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            pt1 = (x1,y1)
            pt2 = (x2,y2)
            image = cv2.rectangle(image,pt1,pt2,green,thickness)

    #Save Image to directory
    path_to_disk
    if not os.path.exists(path_to_disk+'saved_images_new'):
        os.makedirs(path_to_disk+'saved_images_new')
    
    if os.path.exists(path_to_disk+'saved_images_new/'+filename+'.jpg'):
        random_number = random()
        cv2.imwrite(path_to_disk+'saved_images_new/'+filename+'_'+str(random_number)+'.jpg',image)
    else:
        cv2.imwrite(path_to_disk+'saved_images_new/'+filename+'.jpg',image)


    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    grid = make_grid(image)
    writer.add_image('train_image', grid, n)


    return None

def compute_iou(boxA,boxB,epsilon=1e-5):

    '''
    boxA/B: torch.tensor([x1,y1,x2,y2])
    '''

    #boxA = boxA.numpy()
    #boxB = boxB.numpy()

    #Intersection coordinates
    x1 = max(boxA[0],boxB[0])
    y1 = max(boxA[1],boxB[1])
    x2 = min(boxA[2],boxB[2])
    y2 = min(boxA[3],boxB[3])

    #Intersection area
    w = max(0.0,x2-x1 + 1.0)
    h = max(0.0,y2-y1 + 1.0)

    #No overlap
    if (w<0) or (h<0):
        return 0.0
    
    intersection_area = w * h

    #Union
    area_boxa = (boxA[2]-boxA[0] +1.0) * (boxA[3]-boxA[1] + 1.0)
    area_boxb = (boxB[2]-boxB[0] +1.0) * (boxB[3]-boxB[1] +1.0)
    area_union = area_boxa + area_boxb - intersection_area

    iou = intersection_area / (area_union + epsilon)

    return iou

'''def acc_test(gt_boxes,gt_classnames,pd_boxes,pd_classnames):

    accu = 0
    n = 0

    threshold = 0.7

    if len(pd_boxes) == 0:
        print('No boxes detected - from assign')
        return (0,1)
    
    for i,gtb in enumerate(gt_boxes):
        for j,pdb in enumerate(pd_boxes):
            x1y1, x2y2 = pdb
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            pbox = [x1,y1,x2,y2]
            #Change assignment to max iou
            if compute_iou(gtb,pbox) > threshold:
                if gt_classnames[i]==pd_classnames[j]:
                    accu +=1
        n+=1
    
    return (accu,n)'''

def assign_uids(gt_boxes,uids,pd_boxes,embeddings,ed):

    '''
    gt_boxes: list([x1,y1,x2,y2])
    pd_boxes: for box in predicted_boxes:
            x1y1, x2y2 = box
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
    uids = list(ints)
    '''
    threshold = 0.5
    '''print('get box',gt_boxes)
    print('l gtbox',len(gt_boxes))
    print('pd box',pd_boxes)
    print('l pdb',len(pd_boxes))
    sys.exit(0)'''

    if len(pd_boxes) == 0:
        #print('No boxes detected - from assign')
        return len(pd_boxes),len(gt_boxes)

    #Uids are assigned to the predicted box with iou above threshold #T1 -0.62 T5-0.73
    '''for i,gtb in enumerate(gt_boxes):
        for j,pdb in enumerate(pd_boxes):
            x1y1, x2y2 = pdb
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            pbox = [x1,y1,x2,y2]
            #Change assignment to max iou
            if compute_iou(gtb,pbox) > threshold:
                emb = embeddings[j].view(1,-1).numpy()
                uid = np.array(uids[i]).reshape(-1)
                #print('uid',uid)
                #print('embeddings',emb.shape,uid.shape)
                #print('embedding dtype',emb.dtype,uid.dtype)
                ed.addIndex(emb,uid)
                #print('embedding added')
    '''
    
    '''
    for j,pdb in enumerate(pd_boxes):
        for i,gtb in enumerate(gt_boxes):
            x1y1, x2y2 = pdb
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            pbox = [x1,y1,x2,y2]
            #Change assignment to max iou
            if compute_iou(gtb,pbox) > threshold:
                emb = embeddings[j].view(1,-1).numpy()
                uid = np.array(uids[i]).reshape(-1)
                #print('uid',uid)
                #print('embeddings',emb.shape,uid.shape)
                #print('embedding dtype',emb.dtype,uid.dtype)
                ed.addIndex(emb,uid)
                print('embedding added')
    '''

    #Uids are assigned to the box with max iou and above threshold # T1-0.28 T5-0.73
    '''for i,gtb in enumerate(gt_boxes):
        max_iou = 0
        max_embedding_index = 0
        #max_box = 0
        for j,pdb in enumerate(pd_boxes):
            x1y1, x2y2 = pdb
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            pbox = [x1,y1,x2,y2]

            iou = compute_iou(gtb,pbox)
            print('iou',iou)
            if (iou > threshold) and (iou > max_iou):
                max_iou=iou
                max_embedding_index=j
                #max_box=pbox

        emb = embeddings[max_embedding_index].view(1,-1).numpy()
        uid = np.array(uids[i]).reshape(-1)
        ed.addIndex(emb,uid)'''

    #Uids are assigned to the box with max iou #T1-0.46 t5-0.84
    '''for i,gtb in enumerate(gt_boxes):
        #max_iou = 0
        embedding_index = []

        for j,pdb in enumerate(pd_boxes):
            x1y1, x2y2 = pdb
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            pbox = [x1,y1,x2,y2]

            iou = compute_iou(gtb,pbox)
            embedding_index.append(iou)
        
        embedding_index = np.array(embedding_index)
        max_embedding_index = np.argmax(embedding_index)
        emb = embeddings[max_embedding_index].view(1,-1).numpy()
        uid = np.array(uids[i]).reshape(-1)
        ed.addIndex(emb,uid)'''

        #Uids are assigned to the box with the max IoU
    '''
    for j,pdb in enumerate(pd_boxes):
        embedding_index = []

        for i, gtb in enumerate(gt_boxes):
            x1y1, x2y2 = pdb
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            pbox = [x1,y1,x2,y2]

            iou = compute_iou(gtb,pbox)
            embedding_index.append(iou)
        
        print(embedding_index)
        
        embedding_index = np.array(embedding_index)
        max_embedding_index = np.argmax(embedding_index)
        print('max',max_embedding_index)
        print('len embedding',len(embeddings))
        print('len gt,uids',len(gt_boxes),len(uids))
        emb = embeddings[j].view(1,-1).numpy()
        uid = np.array(uids[max_embedding_index]).reshape(-1)
        ed.addIndex(emb,uid)
        #print(emb.shape)
        #print(uid.shape)
        print('embedding added') # emb shape (1,64) uid shape (1,)
    '''

    '''print('STATS')
    print('gtb',len(gt_boxes))
    print('uids',len(uids))
    print('pdbs',len(pd_boxes))
    print('emds',len(embeddings))'''
    

    #Uids are assigned to the box with max iou and above threshold # T1-0.28 T5-0.73
    '''for j,pdb in enumerate(pd_boxes):
        max_iou = 0
        max_embedding_index = 0
        #max_box = 0
        for i, gtb in enumerate(gt_boxes):
            x1y1, x2y2 = pdb
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            pbox = [x1,y1,x2,y2]

            iou = compute_iou(gtb,pbox)
            #print('iou',iou)
            if (iou >= threshold) and (iou > max_iou):
                max_iou=iou
                max_embedding_index=i
                #max_box=pbox

        emb = embeddings[j].view(1,-1).numpy()
        uid = np.array(uids[max_embedding_index]).reshape(-1)
        ed.addIndex(emb,uid)'''

    '''gtboxes = list(gt_boxes)
    for i, pdb in enumerate(pd_boxes):

        if len(gtboxes)==0:
            return len(pd_boxes),len(gt_boxes)

        iou_gtb = []
        #uid_gtb = []

        for j,gtb in enumerate(gtboxes):

            x1y1, x2y2 = pdb
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            pbox = [x1,y1,x2,y2]

            iou = compute_iou(gtb,pbox)
            iou_gtb.append(iou)

            #uid_gtb.append(uids[j])
        
        iou_gtb = np.array(iou_gtb)
        max_iou_index = np.argmax(iou_gtb)

        if iou_gtb[max_iou_index] >=threshold: 
            emb = embeddings[i].view(1,-1).numpy()
            uid = np.array(uids[max_iou_index]).reshape(-1)
            ed.addIndex(emb,uid)
            #print('ADDED: ---->',uid)

            #Removing the matched ground-truth
            gtboxes.remove(gtboxes[max_iou_index])
            uids.remove(uids[max_iou_index])'''

    gtboxes = list(gt_boxes)
    for i,pdb in enumerate(pd_boxes):

        x1y1, x2y2 = pdb
        x1, y1 = int(x1y1[0]), int(x1y1[1])
        x2, y2 = int(x2y2[0]), int(x2y2[1])
        pbox = [x1,y1,x2,y2]

        if len(gt_boxes)==0:
            return len(pd_boxes),len(gt_boxes)

        for j,gtb in enumerate(gtboxes):

            iou = compute_iou(pbox,gtb)
            if iou >=threshold:
                emb = embeddings[i].view(1,-1).numpy()
                uid = np.array(uids[j]).reshape(-1)
                ed.addIndex(emb,uid)

                #Removing the matched ground-truth
                gtboxes.remove(gtb)
                uids.remove(uids[j])
            
    
    '''print('UID',uid)
        ruid = ed.getID(emb,'top1')
        print('ruid',ruid)
        '''

    
    return len(pd_boxes),len(gt_boxes)

def objects_in_tensor(target_tensor):

    class_mask_target = (target_tensor[:,:,:,4]==1) & (target_tensor[:,:,:,9]==1)
    target_tensor_objects = target_tensor[class_mask_target]
   
    return target_tensor_objects.size(0)   


        
def new_validate(loader, detector, ed,writer,uids_list):

    '''
        preds: (dict) {class_name_1: [[filename, prob, x1, y1, x2, y2], ...], class_name_2: [[], ...], ...}.
        targets: (dict) {(filename, class_name): [[x1, y1, x2, y2], ...], ...}.
    '''


    #Switch to evaluate mode
    #model.eval()
    #locEm = locEmDetector(model_path, gpu_id=gpu_id, conf_thresh=-1.0, prob_thresh=-1.0, nms_thresh=0.45)

    '''preds: (dict) {class_name_1: [[filename, prob, x1, y1, x2, y2], ...], class_name_2: [[], ...], ...}.
        targets: (dict) {(filename, class_name): [[x1, y1, x2, y2], ...], ...}.'''

    targets_ev = defaultdict(list)
    preds_ev = defaultdict(list)

    targets_ev_uid = defaultdict(list)
    preds_ev_uid = defaultdict(list)

    unique_keys = pd.read_pickle("../data/unique_keys.pkl")

    map_vid = pd.read_pickle("../data/map_vid.pkl")
    map_cat = map_vid.to_dict()['category_name']
    class_dict = {map_cat[i] for i in map_cat}

    accurate_class_predictions = 0
    total_predictions = 0
    same_len_buffer = 0
    no_pred = 0
    n=0

    #accuracy = 0
    #n_objects = 0 

    correct_samples_ac1 = 0
    correct_samples_ac5 = 0
    objects_target = 0

    print('Len of loader',len(loader))
    t_pd_boxes = 0
    t_gt_boxes = 0

    with torch.no_grad():
        for i, (image,bbox,classname,filename, uids,class_ids,target_tensor) in enumerate(loader):
            '''
                images = tensor(1,3,224,224)
                target = tensor(idx) #idx of dval
            '''
        
            ''' print('image',image.shape)
            print('bbox',bbox)
            print('classname',classname)
            print('filename',filename)
            
            
            print("TYPES")

            print('image',type(image))
            print('bbox',type(bbox[0]))
            print('classname',type(classname[0]))
            print('filename',type(filename))'''
            
            for b in range(len(bbox)):

                x1,y1,x2,y2 = bbox[b]
                targets_ev[(filename,classname[b])].append([x1,y1,x2,y2])

            '''print('classname_t',classname)
            print('bbox_t',bbox)
            print('filename',filename)
            '''
            #preds_ev[class_name].append([sample.file, 0.99, x1, y1, x2, y2])
            boxes, class_names, probs, embeddings_detected, pred_tensor = detector.detect(image)
            '''print('TESTING OUTPUT vs TARGET')
            print(class_names,boxes)
            print('------------Target below-------')
            print(classname,[x1,y1,x2,y2])
            sys.exit(0)'''
            '''print('boxes',len(boxes))
            print('uids',uids)
            print('cnames',len(class_names))
            print('embedd-det',len(embeddings_detected))
            sys.exit(0)'''
            from debug import verify
            #verify(pred_tensor,target_tensor)
            


            if not isinstance(pred_tensor,list):
                
                res = class_decoder(pred_tensor,target_tensor,accuracy)
                #print('res',type(res),res)
                ac1_res,ac5_res =res
                #print('correct',type(correct),correct)
                #print('nobjs',type(nobjs),nobjs)
                correct_samples_ac1+=ac1_res[0]
                correct_samples_ac5+=ac5_res[0]
                objects_target+=ac1_res[1]
            else:
                images_noobjs = objects_in_tensor(target_tensor)
                objects_target+=images_noobjs


            '''a1,a5 = class_decoder(pred_tensor,encoded_target,accuracy)
            acc1+=a1/100.0
            acc5+=a5/100.0'''

            '''ac,no = acc_test(bbox,classname,boxes,class_names)
            accuracy+=ac
            n_objects+=n_objects'''

            '''print("Detected")
            print('boxes',boxes)
            print('class_names',class_names)
            print('probs',probs)
            print('embeddings_detected size',len(embeddings_detected))
            sys.exit(0)'''

            '''print('bbox',bbox)
            print('len bbox',len(bbox))
            print('boxes',boxes)'''
            
            #Assigning uids and saving embeddings to ed
            '''print('uids',type(uids),len(uids))
            print('class_id',type(class_ids),len(class_ids))
            sys.exit(0)'''

            pdb,gtb = assign_uids(bbox,uids,boxes,embeddings_detected,ed)
            t_pd_boxes+=pdb
            t_gt_boxes+=gtb
            #assign_uids(bbox,uids,boxes,embeddings_detected,ed)
            #print('Finished assiging ids')

            '''if len(embeddings_detected) == len(uids):
                same_len_buffer+=1'''

            if len(boxes) == len(bbox):
                same_len_buffer+=1

            if len(boxes)==0:
                no_pred+=1
                

            #total_predictions+=1
            #if len(class_names) > 0 and set(classname) == set(class_names): #set is not an accurate measure needs to be changed
                #accurate_class_predictions+=1
            
            ''' print('class_names',class_names)
            print('boxes',boxes)
            sys.exit(0)'''
            if True:
                
                #visualize(image,bbox,boxes,writer,i,filename)
                n+=1

            #sys.exit(0)

            
            for box, classname_p, prob in zip(boxes, class_names, probs):
                x1y1, x2y2 = box
                x1, y1 = int(x1y1[0]), int(x1y1[1])
                x2, y2 = int(x2y2[0]), int(x2y2[1])
                '''if (x1>448) or (y1>448) or (x2>448) or (y2>448):
                    print('pd',filename,classname_p,box)
                    print('pd',bbox)
                    sys.exit(0)'''
                preds_ev[classname_p].append([filename, prob, x1, y1, x2, y2])

        '''#Unique ID Retreieval
        for i, (image,bbox,classname,filename, uids,class_ids,target_tensor) in enumerate(val_loader):
            for b in range(len(bbox)):

                x1,y1,x2,y2 = bbox[b]
                targets_ev_uid[(filename,uids[b])].append([x1,y1,x2,y2])

            print('uids_gt',uids)
            
            boxes, class_names, probs, embeddings_detected, pred_tensor = detector.detect(image)

            if len(boxes)!=0:

                matched_uids = ed.getIDs(embeddings_detected,'top1')
                print('uids_pd',matched_uids)
                sys.exit(0)

                for box, matched_uid, prob in zip(boxes, matched_uids, probs):
                    x1y1, x2y2 = box
                    x1, y1 = int(x1y1[0]), int(x1y1[1])
                    x2, y2 = int(x2y2[0]), int(x2y2[1])
                    preds_ev_uid[matched_uid].append([filename, prob, x1, y1, x2, y2])'''

        '''#Unique ID Retreieval
        for i, (image,bbox,classname,filename, uids,class_ids,target_tensor) in enumerate(loader):
            for b in range(len(bbox)):

                x1,y1,x2,y2 = bbox[b]
                targets_ev_uid[(filename,uids[b])].append([x1,y1,x2,y2])

            #print('uids_gt',uids)
            #print('bbox',len(bbox))
            
            boxes, class_names, probs, embeddings_detected, pred_tensor = detector.detect(image)


            if len(boxes)!=0:

                t0_uids,t5_matched_uids = ed.getIDs(embeddings_detected,'top0-5')

                #print('matched_uids',matched_uids)
                #print('boxes',len(boxes))
                #print('embeddings_detected',len(embeddings_detected))


                for box, class_names_p, prob, t0, t5_matched_uid in zip(boxes, class_names, probs, t0_uids, t5_matched_uids):
                    x1y1, x2y2 = box
                    x1, y1 = int(x1y1[0]), int(x1y1[1])
                    x2, y2 = int(x2y2[0]), int(x2y2[1])
                    preds_ev_uid[t0].append([filename, prob, x1, y1, x2, y2,t5_matched_uid])'''
        
        

    #print('ACCURACY CLASS',(accurate_class_predictions*100.0)/total_predictions)
    print('Evaluate the detection result...')
    #print('same_len_buffer', (same_len_buffer*100.0)/total_predictions)
    #print('no_pred',(no_pred*100.0)/total_predictions)
    #print('total TARGETS_ev len',len(targets_ev))

    print('Object based accuracy - Top1',(correct_samples_ac1*100.0)/objects_target)
    print('Object based accuracy - Top5',(correct_samples_ac5*100.0)/objects_target)

    print('Total PD Boxes',t_pd_boxes)
    print('Total GT Boxes',t_gt_boxes)

    #print('IoU based accuracy',(accuracy*100.0)/n_objects)
    #print('acc1,acc5',acc1,acc5)

    

    #aps = evaluate_retrieval(preds_ev_uid,targets_ev_uid,uid_names=uids_list)
    

    aps = evaluate(preds_ev, targets_ev, class_names=list(class_dict))
    #aps = evaluate(preds_ev_uid, targets_ev_uid, class_names=uids_list)
    #aps = evaluate_uid(preds_ev_uid,targets_ev_uid,class_names=list(class_dict))

    return aps

'''def new_new_validate(val_loader, detector, mode):

    if mode=='train':
        data = pd.read_pickle("../data/metadata_imgnet_vid_train.pkl")
    else:
        data = pd.read_pickle("../data/metadata_imgnet_vid_val_n2.pkl")

    targets_ev = defaultdict(list)
    preds_ev = defaultdict(list)

    map_vid = pd.read_pickle("../data/map_vid.pkl")
    map_cat = map_vid.to_dict()['category_name']
    class_dict = {map_cat[i] for i in map_cat}

    new_aps = defaultdict()

    with torch.no_grad():
        for i, (image,target) in enumerate(val_loader):
            
            #images = tensor(1,3,224,224)
            #target = tensor(idx) #idx of dval
            
            targets_ev = defaultdict(list)
            preds_ev = defaultdict(list)

            idx = target.item()
            
            sample = data.loc[idx]
            class_name = map_cat[sample.cat_code-1]
            x1,y1,x2,y2 = sample.xmin,sample.ymin,sample.xmax,sample.ymax
            x1,y1,x2,y2 = rescaleBoundingBox(sample.height,sample.width,224,x2,x1,y2,y1)
            targets_ev[(sample.file,class_name)].append([x1,y1,x2,y2])

            boxes, class_names, probs, embeddings_detected = detector.detect(image)
            for box, classname, prob in zip(boxes, class_names, probs):
                x1y1, x2y2 = box
                x1, y1 = int(x1y1[0]), int(x1y1[1])
                x2, y2 = int(x2y2[0]), int(x2y2[1])
                preds_ev[classname].append([sample.file, prob, x1, y1, x2, y2])

            aps_dict = evaluate(preds_ev, targets_ev, class_names=list(class_dict))

            for class_name in aps_dict:
                new_aps[class_name].append(aps_dict[class_name])
    
    #new_aps = dict {'c0':[v1,v1....vn]}
    output_aps = {}
    for class_name in new_aps:
        output_aps[class_name] = sum(new_aps[class_name])

    return output_aps'''



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

def class_decoder(pred_tensor,target_tensor,accuracy):

    '''
        Args:
        Out:
    '''
    #Alternate class_mask_target
    class_mask_target = (target_tensor[:,:,:,4]==1) & (target_tensor[:,:,:,9]==1)

    #class_mask_target tensor is used to identify the gamma boxes in pred_tensor
    pred_tensor_gamma = pred_tensor[class_mask_target]
    pred_tensor_gamma = pred_tensor_gamma[:,B*X:B*X+C] #We only want the class embeddings [n_objects,C]

    #class_mask_target tensor is used to identify the gamma boxes in target_tensor
    target_tensor_gamma = target_tensor[class_mask_target]
    target_tensor_gamma = target_tensor_gamma[:,B*X:B*X+C] #We only want the class embeddings [n_objects,C]

    #Finds the class label
    target = torch.argmax(target_tensor_gamma, dim=1)  #[n_objects,1]
    target = target.view(-1,1)
    output = pred_tensor_gamma #[n_objects,C]

    correct,batch_size = accuracy(output, target, topk=(1, 5))


    return correct,batch_size

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
            #res.append(correct_k.mul_(100.0 / batch_size))
            res.append((correct_k.item(),batch_size))
        return res


if __name__ == '__main__':
    main()
