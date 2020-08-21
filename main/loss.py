import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import traceback
import warnings
import sys


class locemLoss(nn.Module):

    def __init__(self, feature_size=7, num_bboxes=2, num_classes=30, lambda_coord=5.0, lambda_noobj=0.5,beta=64,gamma=1):

        """ Constructor.
        Args:
            feature_size: (int) size of input feature map.
            num_bboxes: (int) number of bboxes per each cell.
            num_classes: (int) number of the object classes.
            lambda_coord: (float) weight for bbox location/size losses.
            lambda_noobj: (float) weight for no-objectness loss.
        """
        super(locemLoss, self).__init__()

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

    def compute_iou(self, bbox1, bbox2):
        '''
        Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        '''

        '''print('bbox1 size',bbox1.size())
        print('bbox2 size',bbox2.size())
        print('bbox1',bbox1)
        print('bbox2',bbox2)
        sys.exit(0)'''


        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou

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
        N = (5 * B) + C    # 5=len([x, y, w, h, conf]


        batch_size = pred_tensor.size(0)

        #Are we considering only one of the targetd boxes, it should be fine since both have same values and confidence in targer
        anchor_mask = (target_tensor[:,:,:,N] == 1) & (target_tensor[:,:,:,4] > 0)
        positive_mask = (target_tensor[:,:,:,N] == 2) & (target_tensor[:,:,:,4] > 0)
        negative_mask = (target_tensor[:,:,:,N] == 3) & (target_tensor[:,:,:,4] > 0)

        #The non-triplet mask can contain non-triplet embeddings or background embeddings
        ##non_triplet_mask = (target_tensor[:,:,:,N] != 1) & (target_tensor[:,:,:,N] != 2) & (target_tensor[:,:,:,N] != 3) #These conditions are enough to include non-triplet objects and/or background embeddings
        ##non_triplet_pred = pred_tensor[non_triplet_mask]
        #CHECK! non_triplet_pred might need to be the same size as anchor and positive
        ##if non_triplet_pred.size(0)!=0:
            ##non_triplets = non_triplet_pred[:,N:]
            ##triplet_loss_bgrnd = nn.TripletMarginLoss(margin=1.0,p=2,reduction='mean')
            ##loss_triplet_bgrnd = triplet_loss_bgrnd(anchor,positive,non_triplets) 


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

        '''print("INIT_TRIPLET_LOSS",loss_triplet)
        print("INIT_TRIPLET_LOSS SIZE",loss_triplet.size())'''

        #NON-TRIPLET EMBEDDINGS
        '''non_triplet_mask = target_tensor[:,:,:,40] & target_tensor[:,:,:,4] == 0
        non_triplet_mask = non_triplet_mask.unsqueeze(-1).expand_as(pred_tensor)
        non_triplets = pred_tensor[non_triplet_mask].view(-1,N+beta)'''

        '''print("PRED TENSOR",pred_tensor.size())
        print('TARGET TENSOR',target_tensor.size())'''

        mode=1 #mode 0: box/class loss for non-triplet objects; 1: box/class loss for ONLY triplet-objects

        #if mode==0:
        pred_tensor = pred_tensor[:,:,:,:N]
        target_tensor = target_tensor[:,:,:,:N]
        coord_mask = target_tensor[:,:,:,4] > 0
        noobj_mask = target_tensor[:,:,:,4] == 0
        '''else:
            #Do we need to filter out the triplet boxes
            ##coord_mask = target_tensor[:,:,:,4] > 0 & (target_tensor[0,:,:,40] == 1 | target_tensor[0,:,:,40] == 2 | target_tensor[0,:,:,40] == 3)
            ##noobj_mask = target_tensor[:,:,:,4] == 0 & (target_tensor[0,:,:,40] == 1 | target_tensor[0,:,:,40] == 2 | target_tensor[0,:,:,40] == 3)
            coord_mask = target_tensor[:, :, :, 4] > 0  # mask for the cells which contain objects. [n_batch, S, S]
            noobj_mask = target_tensor[:, :, :, 4] == 0 # mask for the cells which do not contain objects. [n_batch, S, S]
            pred_tensor = pred_tensor[:,:,:,:N]
            target_tensor = target_tensor[:,:,:,:N]'''

        '''print("PRED TENSOR",pred_tensor.size())
        print('TARGET TENSOR',target_tensor.size())'''
    
        
        #The Yolo Part

        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]

        coord_pred = pred_tensor[coord_mask].view(-1, N)            # pred tensor on the cells which contain objects. [n_coord, N]
                                                                    # n_coord: number of the cells which contain objects.
        bbox_pred = coord_pred[:, :5*B].contiguous().view(-1, 5)    # [n_coord x B, 5=len([x, y, w, h, conf])]
        class_pred = coord_pred[:, 5*B:] 
        
        #print('bbox_pred',bbox_pred)                          # [n_coord, C]

        coord_target = target_tensor[coord_mask].view(-1, N)        # target tensor on the cells which contain objects. [n_coord, N]
                                                                    # n_coord: number of the cells which contain objects.
        bbox_target = coord_target[:, :5*B].contiguous().view(-1, 5)# [n_coord x B, 5=len([x, y, w, h, conf])]
        class_target = coord_target[:, 5*B:]                     # [n_coord, C] 
        #print('bbox_target',bbox_target)


        # Compute loss for the cells with no object bbox.
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)        # pred tensor on the cells which do not contain objects. [n_noobj, N]
                                                                # n_noobj: number of the cells which do not contain objects.
        noobj_target = target_tensor[noobj_mask].view(-1, N)    # target tensor on the cells which do not contain objects. [n_noobj, N]
                                                                # n_noobj: number of the cells which do not contain objects.

        #cuda = torch.cuda.current_device()

        #noobj_conf_mask = torch.cuda.ByteTensor(noobj_pred.size()).fill_(0) # [n_noobj, N] 
        noobj_conf_mask = torch.zeros(noobj_pred.size(), dtype=torch.bool).cuda() # [n_noobj, N] 
        for b in range(B):
            #noobj_conf_mask[:, 4 + b*5] = 1
            noobj_conf_mask[:, 4 + b*5] = True                     # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1 
        noobj_pred_conf = noobj_pred[noobj_conf_mask]           # [n_noobj, 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask]       # [n_noobj, 2=len([conf1, conf2])]
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for the cells with objects.
        #coord_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(0)    # [n_coord x B, 5]
        coord_response_mask = torch.zeros(bbox_target.size(), dtype=torch.bool).cuda()    # [n_coord x B, 5]
        #coord_not_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(1)# [n_coord x B, 5]
        coord_not_response_mask = torch.ones(bbox_target.size(), dtype=torch.bool).cuda() # [n_coord x B, 5]
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()                    # [n_coord x B, 5], only the last 1=(conf,) is used

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):

            pred = bbox_pred[i:i+B] # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = Variable(torch.FloatTensor(pred.size())) # [B, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            pred_xyxy[:,  :2] = pred[:, :2]/float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2]/float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i] # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
            target = bbox_target[i].view(-1, 5) # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            target_xyxy = Variable(torch.FloatTensor(target.size())) # [1, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            target_xyxy[:,  :2] = target[:, :2]/float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2]/float(S) + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4]) # [B, 1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i+max_index] = 1
            coord_not_response_mask[i+max_index] = 0

            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            bbox_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()

        bbox_target_iou = Variable(bbox_target_iou).cuda()

        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)      # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)        # [n_response, 5], only the last 1=(conf,) is used

        '''print('bbox_pred_response',bbox_pred_response[:, 2:4])
        print('bbox_target_response',bbox_target_response[:, 2:4])'''

        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        
        #Handling negative square root loss
        loss_wh = F.mse_loss(
            torch.sign(bbox_target_response[:, 2:4])*torch.sqrt(torch.abs(bbox_target_response[:, 2:4])+1e-8),
            torch.sign(bbox_pred_response[:, 2:4])*torch.sqrt(torch.abs(bbox_pred_response[:, 2:4])+1e-8),
            reduction='sum'
        )
        
        #loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')
        #loss_class = F.nll_loss(class_pred, class_target, reduction='sum')

        #loss_class = 0.7 * loss_class
        #loss_triplet = 0.3 * loss_triplet

        # Total loss
        #CHEK! Should triplet_loss be divided by the number of samples?
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class 
        loss = (loss / float(batch_size))+ loss_triplet

        
        '''print('XY LOSS',loss_xy)
        print('WH LOSS',loss_wh)
        print('OBJ LOSS',loss_obj)
        print('NOBJ LOSS',loss_noobj)
        print('CLASS LOSS',loss_class)
        print('LOSS TRIPLET TYPE',loss_triplet)
        print('LOSS',loss)'''

        loss_class = loss_class/float(batch_size)
        loss_boxes = (self.lambda_coord * (loss_xy + loss_wh))/float(batch_size)
        
        '''print('loss_class',loss_class)
        print('loss_obj',loss_obj + self.lambda_noobj * loss_noobj)
        print('loss_boxes',loss_boxes)
        print('loss_triplet',loss_triplet)
        import sys
        sys.exit(0)'''

        return loss, loss_class, loss_triplet, loss_boxes
