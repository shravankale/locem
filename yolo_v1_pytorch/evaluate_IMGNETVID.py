import argparse
import os, sys
import random
import shutil
import time
import warnings
import numpy as np
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


#For LocEm
from loss import Loss
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
from main.genINV_Locem_Eval_v2 import ImageNetVID
from detect import yoloDetector
from r50_locem import resnet50

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
parser.add_argument('-ep','--experiment_path', type=str, help="Name of the experiment that contains the best model")
#parser.add_argument('-pd','--path_to_disk',default='/disk/shravank/imageNet_ResNet50_savedModel/', type=str, help="Path to disk")


best_acc1 = 0
path_to_disk = '/disk/shravank/results/locem/main/run/'

S=7
B=2
X=5
C=30
image_size = 448

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
         model = resnet50(pretrained=True,S=S,B=B,C=C,X=X)
    else:
        print("=> creating model '{}'".format(args.arch))
        #model = models.__dict__[args.arch]()
        model = resnet50(S=S,B=B,C=C,X=X)

    num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, final_layer_units)

    #SIGMOID WAS ADDED BECAUSE SOME OF THE PREDICTED VALUES WERE NEGATIVE
    #DURING WH LOSS CALCULATION WHICH TAKES SQUARE ROOT OF WH, NANs WERE INTRODUCED INTO THE LOSS
    #BUT SOME PREDICTED VALUES MIGHT NEED TO BE NEGATIVE
    #https://www.reddit.com/r/deeplearning/comments/9z50qi/confused_about_yolo_loss_function/

    '''
    num_classes = S*S*(B*X+C)
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs,4096),
        #nn.LeakyReLU(0.1, inplace=True),
        nn.ReLU(),
        #nn.Dropout(0.5, inplace=False),
        nn.Linear(4096,num_classes),
        nn.Sigmoid(),
        View((-1,S,S,B*X+C))
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
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = Loss(eature_size=S, num_bboxes=B, num_classes=C, lambda_coord=5.0, lambda_noobj=0.5)

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
    root_datasets = '/disk/shravank/datasets/'


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

    writer = SummaryWriter(path_to_disk)

    #detector = yoloDetector(args.experiment_path)
    detector = locEmDetector(model,conf_thresh=0.2, prob_thresh=0.2, nms_thresh=0.60,S=S,B=B,C=C,X=X,image_size=image_size)
    aps = new_validate(val_loader, detector,writer)

    '''print('Mean APS',np.mean(aps))

    map_vid = pd.read_pickle("../../data/map_vid.pkl")
    map_cat = map_vid.to_dict()['category_name']
    #dval = pd.read_pickle("../data/metadata_imgnet_vid_val_n2.pkl")
    class_dict = {map_cat[i] for i in map_cat}
    
    class_aps_dict = {}
    for i,j in zip(class_dict,aps):
        class_aps_dict[i]=j
    print(class_aps_dict)'''

    #writer = SummaryWriter(path_to_disk)

    return


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
       
def evaluate(preds,targets,class_names,threshold=0.5):
    
    """ Compute mAP metric.
    Args:
        preds: (dict) {class_name_1: [[filename, prob, x1, y1, x2, y2], ...], class_name_2: [[], ...], ...}.
        targets: (dict) {(filename, class_name): [[x1, y1, x2, y2], ...], ...}.
        class_names: (list) list of class names.
        threshold: (float) threshold for IoU to separate TP from FP.
    Returns:
        (list of float) list of average precision (AP) for each class.
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

def visualize(image,target_boxes,predicted_boxes,writer,n):

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

    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    grid = make_grid(image)
    writer.add_image('train_image', grid, n)


    return None
        
def new_validate(val_loader, detector):


    #Switch to evaluate mode
    #model.eval()
    #locEm = yoloDetector(model_path, gpu_id=gpu_id, conf_thresh=-1.0, prob_thresh=-1.0, nms_thresh=0.45)

    targets_ev = defaultdict(list)
    preds_ev = defaultdict(list)

    map_vid = pd.read_pickle("../data/map_vid.pkl")
    map_cat = map_vid.to_dict()['category_name']
    class_dict = {map_cat[i] for i in map_cat}

    accurate_class_predictions = 0
    total_predictions = 0
    n=0

    with torch.no_grad():
        for i, (image,bbox,classname,filename,uids) in enumerate(val_loader):
            '''
                images = tensor(1,3,224,224)
                target = tensor(idx) #idx of dval
            '''

            #print("TYPE imAGE",type(image))
            #Create target

            '''idx = target.item()
            
            sample = data.loc[idx]
            class_name = map_cat[sample.cat_code-1]
            x1,y1,x2,y2 = sample.xmin,sample.ymin,sample.xmax,sample.ymax'''
            '''x1,y1,x2,y2 = bbox
            targets_ev[(filename,classname)].append([x1,y1,x2,y2])'''

            for b in range(len(bbox)):

                x1,y1,x2,y2 = bbox[b]
                targets_ev[(filename,classname[b])].append([x1,y1,x2,y2])

            #preds_ev[class_name].append([sample.file, 0.99, x1, y1, x2, y2])
            boxes, class_names, probs = detector.detect(image)
            '''print('TESTING OUTPUT vs TARGET')
            print(class_names,boxes)
            print('------------Target below-------')
            print(classname,[x1,y1,x2,y2])
            sys.exit(0)'''
            total_predictions+=1
            if len(class_names) > 0 and set(classname) == set(class_names): #set is not an accurate measure needs to be changed
                accurate_class_predictions+=1

            if True:
                
                visualize(image,bbox,boxes,writer,i)
                n+=1

            for box, classname_p, prob in zip(boxes, class_names, probs):
                x1y1, x2y2 = box
                x1, y1 = int(x1y1[0]), int(x1y1[1])
                x2, y2 = int(x2y2[0]), int(x2y2[1])
                preds_ev[classname].append([filename, prob, x1, y1, x2, y2])

    print('ACCURACY CLASS',(accurate_class_predictions*100.0)/total_predictions)
    print('Evaluate the detection result...')

    aps = evaluate(preds_ev, targets_ev, class_names=list(class_dict))

    return aps

def new_new_validate(val_loader, detector, mode):

    if mode=='train':
        data = pd.read_pickle("../../data/metadata_imgnet_vid_train.pkl")
    else:
        data = pd.read_pickle("../../data/metadata_imgnet_vid_val_n2.pkl")

    targets_ev = defaultdict(list)
    preds_ev = defaultdict(list)

    map_vid = pd.read_pickle("../../data/map_vid.pkl")
    map_cat = map_vid.to_dict()['category_name']
    class_dict = {map_cat[i] for i in map_cat}

    new_aps = defaultdict()

    with torch.no_grad():
        for i, (image,target) in enumerate(val_loader):
            '''
                images = tensor(1,3,224,224)
                target = tensor(idx) #idx of dval
            '''
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

    return output_aps



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



if __name__ == '__main__':
    main()
