
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class locemLoss(nn.Module)

    def __init__(self,S,B,coord,noobj,alpha,beta):
        super().__init__()
        self.S = S
        self.B = B

        #Regularization for co-ordinate loss
        self.coord = coord
        #Regularization for no-object in bounding box loss
        self.noobj = noobj

        #Number of classes
        self.alpha = alpha
        #Length of embedding
        self.beta = beta

    def iou(self,box1,box2)

        '''
        Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
            Args:
            box1: (tensor) bounding boxes, sized [N,4].
            box2: (tensor) bounding boxes, sized [M,4].
            Return:
            (tensor) iou, sized [N,M]
        '''

        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:,:2].unsqueeze(1).expand(N,M,2), #[:,N,2] -> [:,N,1,2] -> [:,N,M,2]
            box2[:,:,:2].unsqueeze(0).expand(N,M,2)  #[:,M,2] -> [:1,M,2] -> [:,N,M,2]
        )
        rb = torch.max(
            box1[:,:,2:].unsqueeze(1).expand(N,M,2), #[:,N,2] -> [:,N,1,2] -> [:,N,M,2]
            box2[:,:,2:].unsqueeze(0).expand(N,M,2)  #[:,M,2] -> [:1,M,2] -> [:,N,M,2]
        )

        wh = rb-lt
        wh[wh<0] = 0 #min value = 0
        intersection = wh[:,:,0] * wh[:,:,1] #[N,M]

        area_b1 = (box1[:,2]) - (box1[:,0]) * (box1[:,3]) - (box1[:,1]) #[N,]
        area_b2 = (box2[:,2]) - (box2[:,0]) * (box2[:,3]) - (box2[:,1]) #[M,]
        area_b1 = area_b1.unsqueeze(1).expand_as(intersection) #[N,] -> [N,1], -> [N,M]
        area_b2 = area_b2.unsqueeze(0).expand_as(intersection) #[M,] -> [1,M] -> [N,M]

        union = area_b1 + area_b2 - intersection
        iou = intersection/union

        return iou

    def forward(self,pred_tensor,target_tensor):

        '''
        pred_tensor size = (batch_size,S,S,B*X + alpha + beta) #(b,S,S,B*5 + 30 + 2048)
        target_tensor size = (batch_size,S,S,B*X + alpha + gamma)

        Where,

        S * S = Grid Size (3,3) or (7,7)
        B = Number of boxes (2)
        X = [xmin,ymin,xmax,ymax,confidence]
        alpha = len(classes) = 30
        beta = len(embedding) = 1024
        gamma = len(embedding_identifier) = 3 #0:non-triplet 1:anchor - 2:positive - 3:negative

        out_tensor = coord(loss_bbox) + (object=1)*conf + (noobj)*conf + 0.5(loss_classification) + 0.5(triplet_loss)       

        '''
        #del
        N = pred_tensor.size()[0]
        X=5
        gamma = 1
        alpha,beta = self.apha,self.beta

        '''anchor = pred_tensor[0,:,:,B*5+30:]
        positive = pred_tensor[1,:,:,B*5+30:]
        negative = pred_tensor[2,:,:,B*5+30:]'''

        #The Triplet masks have appropriate gamma values and their box confidence greater than 0
        #Now the question is should the class and box confidense values be only calculated for the triplet boxes
        anchor_mask = target_tensor[0,:,:,40] == 1 & target_tensor[:,:,:,4] > 0
        positive_mask = target_tensor[1,:,:,40] == 2 & target_tensor[:,:,:,4] > 0
        negative_mask = target_tensor[2,:,:,40] == 3 & target_tensor[:,:,:,4] > 0

        #Now if we are calculating the class and box confidence for only the triplet boxes then can coo_mask be the triplet_mask?


    
        anchor_mask = anchor_mask.unsqueeze(-1).expand_as(pred_tensor)
        positive_mask = positive_mask.unsqueeze(-1).expand_as(pred_tensor)
        negative_mask = negative_mask.unsqueeze(-1).expand_as(pred_tensor)

        #THE TRIPLETS MAY NOT REQUIRES_GRAD=TRUE BECAUSE PRED_TENSOR ALREADY HAS REQUIRES_GRAD=TRUE
        anchor_pred = pred_tensor[anchor_mask].view(-1,B*X+alpha+beta)
        #anchor = torch.Tensor(anchor_pred[:,40:],requires_grad=True)
        anchor = anchor_pred[:,B*X+alpha:]
        positive_pred = pred_tensor[positive_mask].view(-1,B*X+alpha+beta)
        #positive = torch.Tensor(positive_pred[:,40:],requires_grad=True)
        positive = positive_pred[:,B*X+alpha:]
        negative_pred = pred_tensor[negative_mask].view(-1,B*X+alpha+beta)
        #negative = torch.Tensor(negative_pred[:,40:],requires_grad=True)
        negative = negative_pred[:,B*X+alpha:]

        #Check **kwargs
        triplet_loss = nn.TripletMarginLoss(anchor,positive,negative)

        #NON-TRIPLET EMBEDDINGS
        non_triplet_mask = target_tensor[:,:,:,40] & target_tensor[:,:,:,4] == 0
        non_triplet_mask = non_triplet_mask.unsqueeze(-1).expand_as(pred_tensor)
        non_triplets = pred_tensor[non_triplet_mask].view(-1,B*X+alpha+beta)

        #SWITCH PRED_TENSOR and TARGET_TENSOR TO YOLO LIKE tensors of equal length

        mode=0 #mode 0: box/class loss for non-triplet objects; 1: box/class loss for ONLY triplet-objects

        if mode==0:
            pred_tensor = pred_tensor[:,:,:,B*X+alpha]
            target_tensor = target_tensor[:,:,:,B*X+alpha]
            coo_mask = target_tensor[:,:,:,4] > 0
            noo_mask = target_tensor[:,:,:,4] == 0
        else:
            coo_mask = target_tensor[:,:,:,4] > 0 & (target_tensor[0,:,:,40] == 1 | target_tensor[0,:,:,40] == 2 | target_tensor[0,:,:,40] == 3)
            noo_mask = target_tensor[:,:,:,4] == 0 & (target_tensor[0,:,:,40] == 1 | target_tensor[0,:,:,40] == 2 | target_tensor[0,:,:,40] == 3)
            pred_tensor = pred_tensor[:,:,:,B*X+alpha]
            target_tensor = target_tensor[:,:,:,B*X+alpha]

        
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1,30)
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:,10:]                       #[x2,y2,w2,h2,c2]
        
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # compute not contain obj loss 
        noo_pred = pred_tensor[noo_mask].view(-1,30)
        noo_target = target_tensor[noo_mask].view(-1,30)
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)

        #compute contain obj loss
        coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()

        for i in range(0,box_target.size()[0],2): #choose the best iou box
            box1 = box_pred[i:i+2]
            #Since the prediction starts with random numbers, asssume it's in x,y,w,h format
            '''box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2]/14. -0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2]/14. +0.5*box1[:,2:4]'''

            box2 = box_target[i].view(-1,5)
            #Need to convert xc,yc,w,h to x1,y1,x2,y2
            '''box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2]/14. -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2]/14. +0.5*box2[:,2:4]'''

            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            '''#x1 = xc - w/2
            box2_xyxy[:,0] = box2[:,0] - (box[:,2]/2)
            #y1 = yc - h/2
            box2_xyxy[:,1] = box2[:,1] - (box[:,3]/2)
            #x2 = xc + w/2
            box2_xyxy[:,2] = box2[:,0] + (box[:,2]/2)
            #y2 = yc + h/2
            box2_xyxy[:,3] = box2[:,1] + (box[:,3]/2)'''

            #Streamlining above 4 equations
            box2_xyxy[:,:2] = box2[:,:2] -0.5*(box2[:,2:4])
            box2_xyxy[:,2:4] = box2[:,:2] +0.5*(box2[:,2:4])

            '''iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]'''
            iou = self.iou(box1[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()
            
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()

        box_target_iou = Variable(box_target_iou).cuda()
        #1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)

    
        loss_xy = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False)
        loss_wh = F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)

        '''loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(box_pred_response[:,2:4],box_target_response[:,2:4],size_average=False)'''
        #2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4]= 0
        #not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        
        #I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)

        #3.class loss
        class_loss = F.mse_loss(class_pred,class_target,size_average=False)

        return (self.l_coord*(loss_xy+loss_wh) + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss + triplet_loss)/N
