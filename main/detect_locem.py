import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import cv2
import numpy as np

from main.nn_view import View

class locEmDetector():

    def __init__(self,
        model, class_name_list=None, mean_rgb=[122.67891434, 116.66876762, 104.00698793],
        conf_thresh=0.1, prob_thresh=0.1, nms_thresh=0.5,
        gpu_id=2,S=7,B=2,C=30,X=5,beta=64,image_size=224):

        map_vid = pd.read_pickle("../data/map_vid.pkl")
        self.class_name_list = list(map_vid['category_name'])
        print('self.class_name_list',self.class_name_list)

        self.S, self.B, self.C, self.beta = S,B,C,beta

        self.conf_thresh = conf_thresh
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.gpu_id = gpu_id
        self.image_size=image_size

        self.to_tensor = transforms.ToTensor()
        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

        #Fetch locEm model
        #self.loceEm = self.getModel(self.model_path)
        self.loceEm = model
        #Set model to validate
        self.loceEm.eval()
        

    def image2tensorboard(self,val_loader):
        '''
        Either create the val_loader here or fetch it here
        '''


    '''def visualize(self,image,target, boxes, class_names, probs): #From locem.detect()
        #Create Writer to the same path as 
        experiment_path = self.model_path.split('/')
        experiment_path = experiment_path[:len(experiment_path)-1]
        experiment_path = '/'.join(experiment_path) + '/'

        #Pre-process target

        writer = SummaryWriter(experiment_path)

        for box, class_name, prob in zip(boxes, class_names, probs):
            # Draw box on the image.
            left_top, right_bottom = box
            left, top = int(left_top[0]), int(left_top[1])
            right, bottom = int(right_bottom[0]), int(right_bottom[1]) 
            x1,y1,x2,y2 = left, top, bottom, right
    
        return 0'''





    '''def getModel(self,model_path):

        S,B,C,beta=self.S, self.B, self.C, self.beta
        X=5

        print("Loading model from ",model_path)
        model = models.__dict__['resnet50']()
        num_ftrs = model.fc.in_features

        num_classes = S*S*(B*X+C+beta)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs,4096),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(),
            #nn.Dropout(0.5, inplace=False),
            nn.Linear(4096,num_classes),
            nn.Sigmoid(),
            View((-1,S,S,B*X+C+beta))
        )

        model = torch.nn.DataParallel(model).cuda()
        optimizer = torch.optim.SGD(model.parameters(), 0.01,
                                momentum=0.9,
                                weight_decay=1e-4)
        loc = 'cuda:{}'.format(self.gpu_id)
        checkpoint = torch.load(model_path, map_location=loc)

        best_acc1 = checkpoint['best_acc1']
        epch = checkpoint['epoch']
        print("Best Training Accuracy: {} @Epoch: {}".format(best_acc1,epch))

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        return model'''

    def detect(self,img):

        image_size=self.image_size

        '''
        img = tensor(1,3,224,224)

        Return: 
        boxes_detected: (list of tuple) box corner list like [((x1, y1), (x2, y2))_obj1, ...]. Re-scaled for original input image size.
            class_names_detected: (list of str) list of class name for each detected boxe.
            probs_detected: (list of float) list of probability(=confidence x class_score) for each detected box.
        '''
        S,B,C,beta = self.S, self.B, self.C, self.beta

        #Converts Tensor(N=1,C,H,W) to NP(H,W,C)
        #img = img.numpy()[0]

        #print("TYPE IMG",type(img))
        #print("IMG.SHAPE",img.shape)


        #Preprocessing image before detect
        #rint('img.shape',img.shape)
        h,w,_ = img.shape
        img = cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # assuming the model is trained with RGB images.
        img = (img - self.mean) / 255.0
        img = self.to_tensor(img)
        img = img[None, :, :, :]  # [3, image_size, image_size] -> [1, 3, image_size, image_size]
        img = Variable(img)
        img = img.cuda()
        #img = img.to(torch.device('cuda:0'))
        

        

        with torch.no_grad():
            pred_tensor = self.loceEm(img)
        pred_tensor = pred_tensor.cpu().data
        pred_tensor_output = pred_tensor.clone().detach()
        pred_tensor = pred_tensor.squeeze(0) # squeeze batch dimension because we are getting detection only for one image

        boxes_normalized_all, class_labels_all, confidences_all, class_scores_all, embeddings_all = self.decode(pred_tensor)

        #print('detect(), class_labels_all',class_labels_all)
        #print('detect(), boxes_normalized_all',boxes_normalized_all)

        if boxes_normalized_all.size(0) == 0:
            
            return [], [], [], [],[] # if no box found, return empty lists.
        
        # Apply non maximum supression for boxes of each class.
        boxes_normalized, class_labels, probs, embeds = [], [], [], []

        for class_label in range(len(self.class_name_list)): 
            mask = (class_labels_all == class_label)

            if torch.sum(mask) == 0:
                continue # if no box found, skip that class.
            
            boxes_normalized_masked = boxes_normalized_all[mask]
            class_labels_maked = class_labels_all[mask]
            confidences_masked = confidences_all[mask]
            class_scores_masked = class_scores_all[mask]
            embeddings_masked = embeddings_all[mask]

            ids = self.nms(boxes_normalized_masked, confidences_masked)

            boxes_normalized.append(boxes_normalized_masked[ids])
            class_labels.append(class_labels_maked[ids])
            probs.append(confidences_masked[ids] * class_scores_masked[ids])
            embeds.append(embeddings_masked[ids])

        boxes_normalized = torch.cat(boxes_normalized, 0)
        class_labels = torch.cat(class_labels, 0)
        probs = torch.cat(probs, 0)
        embeds = torch.cat(embeds,0)

        # Postprocess for box, labels, probs,embeds.
        boxes_detected, class_names_detected, probs_detected, embeddings_detected = [], [], [], []
        for b in range(boxes_normalized.size(0)):
            box_normalized = boxes_normalized[b]
            class_label = class_labels[b]
            prob = probs[b]
            embed = embeds[b]

            x1, x2 = w * box_normalized[0], w * box_normalized[2] # unnormalize x with image width.
            y1, y2 = h * box_normalized[1], h * box_normalized[3] # unnormalize y with image height.
            boxes_detected.append(((x1, y1), (x2, y2)))

            class_label = int(class_label) # convert from LongTensor to int.
            class_name = self.class_name_list[class_label]
            class_names_detected.append(class_name)

            prob = float(prob) # convert from Tensor to float.
            probs_detected.append(prob)

            embeddings_detected.append(embed)

        return boxes_detected, class_names_detected, probs_detected, embeddings_detected, pred_tensor_output

    
    def nms(self, boxes, scores):
        """ Apply non maximum supression.
        Args:
        Returns:
        """
        threshold = self.nms_thresh

        x1 = boxes[:, 0] # [n,]
        y1 = boxes[:, 1] # [n,]
        x2 = boxes[:, 2] # [n,]
        y2 = boxes[:, 3] # [n,]
        areas = (x2 - x1) * (y2 - y1) # [n,]

        _, ids_sorted = scores.sort(0, descending=True) # [n,]
        ids = []
        while ids_sorted.numel() > 0:
            # Assume `ids_sorted` size is [m,] in the beginning of this iter.

            i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
            ids.append(i)

            if ids_sorted.numel() == 1:
                break # If only one box is left (i.e., no box to supress), break.

            inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i]) # [m-1, ]
            inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i]) # [m-1, ]
            inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i]) # [m-1, ]
            inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i]) # [m-1, ]
            inter_w = (inter_x2 - inter_x1).clamp(min=0) # [m-1, ]
            inter_h = (inter_y2 - inter_y1).clamp(min=0) # [m-1, ]

            inters = inter_w * inter_h # intersections b/w/ box `i` and other boxes, sized [m-1, ].
            unions = areas[i] + areas[ids_sorted[1:]] - inters # unions b/w/ box `i` and other boxes, sized [m-1, ].
            ious = inters / unions # [m-1, ]

            # Remove boxes whose IoU is higher than the threshold.
            ids_keep = (ious <= threshold).nonzero().squeeze() # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
            if ids_keep.numel() == 0:
                break # If no box left, break.
            ids_sorted = ids_sorted[ids_keep+1] # `+1` is needed because `ids_sorted[0] = i`.

        return torch.LongTensor(ids)

        
    
    def decode(self,pred_tensor):
        '''
        Decode tensor into box coordinates, class labels, probs_detected, and embeddings.

        Args:
            pred_tensor: (tensor) tensor to decode sized [S, S, 5 x B + C + beta], 5=(x, y, w, h, conf)
        Returns:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...]. Normalized from 0.0 to 1.0 w.r.t. image width/height, sized [n_boxes, 4].
            labels: (tensor) class labels for each detected boxe, sized [n_boxes,].
            confidences: (tensor) objectness confidences for each detected box, sized [n_boxes,].
            class_scores: (tensor) scores for most likely class for each detected box, sized [n_boxes,].
            embeddings: The embedding for the objects refined by class and box coordinates, sized [n_boxes,1]

        '''
        #print('decode()',pred_tensor.size())

        S, B, C = self.S, self.B, self.C
        boxes, labels, confidences, class_scores, embeddings = [], [], [], [], []

        cell_size = 1.0 / float(S)

        conf = pred_tensor[:, :, 4].unsqueeze(2) # [S, S, 1]
        for b in range(1, B):
            conf = torch.cat((conf, pred_tensor[:, :, 5*b + 4].unsqueeze(2)), 2)
        #conf = conf[:,:,1:] #Removing the duplicate first conf
        conf_mask = conf > self.conf_thresh # [S, S, B]

        for i in range(S):
            for j in range(S):

                class_score, class_label = torch.max(pred_tensor[j, i, 5*B:5*B+C], 0)
                embedding = pred_tensor[j,i,5*B+C:]

                for b in range(B):
                    conf = pred_tensor[j, i, 5*b + 4]
                    prob = conf * class_score
                    if float(prob) < self.prob_thresh:  #If the probability is less than a predifined threshold we ignore the box altogether
                        continue    #CAREFUL! If both boxes are dropped then we might losse class+embedding - No it will check the next box

                    # Compute box corners (x1, y1, x2, y2) from tensor.
                    box = pred_tensor[j, i, 5*b : 5*b + 4] #[j,i,xc,yc,w,h]
                    x0y0_normalized = torch.FloatTensor([i, j]) * cell_size #float(i,j) # cell left-top corner. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    xy_normalized = box[:2] * cell_size + x0y0_normalized   # box center. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    wh_normalized = box[2:] # Box width and height. Normalized from 0.0 to 1.0 w.r.t. image width/height.

                    box_xyxy = torch.FloatTensor(4) # [4,] #[x1,y1,x2,y2]
                    box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized # left-top corner (x1, y1).
                    box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized # right-bottom corner (x2, y2).

                    # Append result to the lists.
                    boxes.append(box_xyxy)
                    labels.append(class_label)
                    confidences.append(conf)
                    class_scores.append(class_score)
                    embeddings.append(embedding)

        if len(boxes) > 0:
            boxes = torch.stack(boxes, 0) # [n_boxes, 4]
            labels = torch.stack(labels, 0)             # [n_boxes, ]
            confidences = torch.stack(confidences, 0)   # [n_boxes, ]
            class_scores = torch.stack(class_scores, 0) # [n_boxes, ]
            embeddings = torch.stack(embeddings,0)      #[n_boxes]

        else:
            # If no box found, return empty tensors.
            boxes = torch.FloatTensor(0, 4)
            labels = torch.LongTensor(0)
            confidences = torch.FloatTensor(0)
            class_scores = torch.FloatTensor(0)
            embeddings = torch.FloatTensor(0)

        return boxes, labels, confidences, class_scores, embeddings


