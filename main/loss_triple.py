import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import traceback
import warnings
import sys


class tripletLoss(nn.Module):

    def __init__(self, feature_size=7, num_bboxes=2, num_classes=30, lambda_coord=5.0, lambda_noobj=0.5,beta=64,gamma=1):

        """ Constructor.
        Args:
            feature_size: (int) size of input feature map.
            num_bboxes: (int) number of bboxes per each cell.
            num_classes: (int) number of the object classes.
            lambda_coord: (float) weight for bbox location/size losses.
            lambda_noobj: (float) weight for no-objectness loss.
        """
        super(tripletLoss, self).__init__()

        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        #C = alpha
        self.beta = beta
        self.gamma = gamma
    
    def warn_with_traceback(self,message, category, filename, lineno, file=None, line=None):

        log = file if hasattr(file,'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    def forward(self, pred_tensor, target_tensor):
        warnings.showwarning = self.warn_with_traceback

        '''
        pred_tensor size = (batch_size,S,S,B*X + C + beta) #(b,S,S,B*5 + 30 + gamma)
        target_tensor size = (batch_size,S,S,B*X + C + gamma)

        Where,

        S * S = Grid Size (3,3) or (7,7)
        B = Number of boxes (2)
        X = [xmin,ymin,xmax,ymax,confidence] = len(5)
        C = len(classes) = 30
        beta = len(embedding) = 64
        gamma = len(embedding_identifier) = 1 #0:non-triplet 1:anchor - 2:positive - 3:negative

        out_tensor = coord(loss_bbox) + (object=1)*conf + (noobj)*conf + 0.5(loss_classification) + 0.5(triplet_loss)       

        '''
        S, B, C = self.S, self.B, self.C
        N = 5 * B + C    # 5=len([x, y, w, h, conf]

        batch_size = pred_tensor.size(0)

        #Are we considering only one of the targetd boxes, it should be fine since both have same values and confidence in targer
        anchor_mask = (target_tensor[:,:,:,N] == 1) & (target_tensor[:,:,:,4] > 0)
        positive_mask = (target_tensor[:,:,:,N] == 2) & (target_tensor[:,:,:,4] > 0)
        negative_mask = (target_tensor[:,:,:,N] == 3) & (target_tensor[:,:,:,4] > 0)

        #New masking
        anchor_pred = pred_tensor[anchor_mask]
        positive_pred = pred_tensor[positive_mask]
        negative_pred = pred_tensor[negative_mask]

        anchor = anchor_pred[:,N:]
        positive = positive_pred[:,N:]
        negative = negative_pred[:,N:]

        #Check **kwargs
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
        loss_triplet = triplet_loss(anchor, positive, negative)

        return loss_triplet
