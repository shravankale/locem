import pandas as pd
import numpy as np
import os, fnmatch, re, cv2, random
import torch
#import imgaug as ia
#import imageio
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torch.utils import data
from torchvision import transforms
from PIL import Image
from util.augmentations import Augment

class ImageNetVID(data.Dataset):


    def __init__(self,root_datasets,path_to_dataset,split,image_size,S,B,C,X,gamma,transform=None):
        
        print('USING v101 of generator')
    
        self.transform = transform
        self.root_datasets = root_datasets
        #self.root_ImageNetVidsDevkit = self.root_datasets+"ImageNetVids/imageNetVidsDevkit.data/"
        #self.root_ImageNetVids = self.root_datasets+"ImageNetVids/imageNetVids.data/"
        #self.root_ImageNetVids = ''
        self.split_mode = split
        self.aug = Augment()


        if self.split_mode == 'train':
            self.path_to_frames=self.root_datasets+"ILSVRC2015/Data/VID/train/"
            self.all_data = pd.read_pickle('../data/metadata_imgnet_vid_train.pkl')
            
        elif self.split_mode == 'val':
            self.path_to_frames= self.root_datasets+"ILSVRC2015/Data/VID/val/"
            self.all_data = pd.read_pickle('../data/metadata_imgnet_vid_val.pkl')
           
        else:
            raise ValueError('Split has to be train or val')

        self.data_set = pd.read_pickle(path_to_dataset)
        self.unique_keys = self.getKeys(pd.DataFrame(self.data_set))
        #self.data_set = self.data_set[:100]

        self.network_dim = image_size
        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

        self.to_tensor = transforms.ToTensor()

        self.S,self.B,self.C,self.X,self.gamma = S,B,C,X,gamma

    def encodeTarget_locem(self,target_bbox,target_class,target_id):
        '''
            target_bbox = [xmax,xmin,ymax,ymin] #~~ [x2,x1,y2,y1]
            target_class = int
            target_id = int
            target_output = (S,S,B*X+C+gama)

        '''
        #Figure out label encoding

        S = self.S
        B = self.B
        X = self.X
        C = self.C
        gamma = self.gamma

        #if target_class.size(0) > 1:
            #raise ValueError('Need to adjust for target_class > 1')

        labels = target_class

        #v1
        #boxes = torch.Tensor(target_bbox).view(1,-1)
        #v101
        boxes = target_bbox
        #print('pre-norm',boxes)
        

        #image_height,image_width = 224,224
        #boxes /= torch.Tensor([[image_width, image_height, image_width, image_height]]).expand_as(boxes) # normalize (x1, y1, x2, y2) w.r.t. image width/height.
        #print('post-norm',boxes)
        #Figure out label encoding
        #labels = [target_class] #type(target_class)=list(int) 

        target = torch.zeros(S, S, B*X + C + gamma)
        cell_size = 1.0 / float(S)
        boxes_wh = boxes[:, 2:] - boxes[:, :2] # width and height for each box, [n, 2]
        boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0 # center x & y for each box, [n, 2]

        for b in range(boxes.size(0)):
            xy, wh, label = boxes_xy[b], boxes_wh[b], int(labels[b])

            ij = (xy / cell_size).ceil() - 1.0
            #print('ij',ij)
            i, j = int(ij[0]), int(ij[1]) # y & x index which represents its location on the grid.
            #if i ==7 or j==7:
                #print('OUT OF BOUNDS: ',i,j,xy)
            x0y0 = ij * cell_size # x & y of the cell left-top corner.
            xy_normalized = (xy - x0y0) / cell_size # x & y of the box on the cell, normalized from 0.0 to 1.0.

            for k in range(B):
                #if target[j, i, s+4    ] == 1.0:
                   #continue 
                s = 5 * k
                target[j, i, s  :s+2] = xy_normalized
                target[j, i, s+2:s+4] = wh
                target[j, i, s+4    ] = 1.0 #Confidence of the bounding box
            
            #Setting class label
            target[j, i, B*X + label] = 1.0

            #Setting gamma label for only the first box which is one of the triplets
            #other boxes will be in a grid with target_id = 0. They won't be picked up while calculating triplet score
            if b==0:
                target[j, i, B*X + C] = target_id

        return target

    def class_decoder(self,pred_tensor,target_tensor,accuracy):

        '''
            Args:
            Out:
        '''

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

        return acc1, acc5

    def getKeys(self,dataset):

        dataset = pd.DataFrame(dataset)
        unique_ids = dataset.drop_duplicates(subset=['cat_code','snip_id','trackid'],keep='first')
        unique_ids = unique_ids.reset_index(drop=True)
        unique_ids = unique_ids.drop(labels=['folder','file','width','height','wnid','xmax','xmin','ymax','ymin'],axis=1)

        return unique_ids

    def reduceDataSet(self,dataset):

        #dataset = self.data_set
        self.reduceCounter+=1

        dset_anc_samples=dataset.drop_duplicates(subset=['cat_code','snip_id','trackid'],keep='first')
        dset_pos_samples=dataset.drop_duplicates(subset=['cat_code','snip_id','trackid'],keep='last')
        dataset = pd.concat([dset_anc_samples,dset_pos_samples],ignore_index=True,verify_integrity=True)
        dataset = dataset.drop_duplicates()
        dataset = dataset.reset_index(drop=True)

        print(self.split_mode+' Dataset reduced to len: ',len(dataset))
        self.reduceCounter +=1

        return dataset
    
    def getOtherObjects(self,sample,train_data,all_data):
    
        boxes = []
        classes = []
        
        other_samples = all_data[(all_data.folder==sample.folder)&(all_data.file==sample.file)]
        #If there is only 1 object in image then there is nothing to add
        if len(other_samples)==1:
            return [],[]
        #Dropping sample from other
        other_samples = other_samples[other_samples.trackid!=sample.trackid]
        
        for idx in other_samples.index:
            row = other_samples.loc[idx]
            boxes.append([row.xmin,row.ymin,row.xmax,row.ymax])
            classes.append([row.cat_code-1])
        
        boxes = torch.tensor(boxes,dtype=torch.float)
        classes = torch.tensor(classes,dtype=torch.float)
        
        return boxes,classes
    
        
    def __len__(self):

        length_data_set = len(self.data_set)
        print(self.split_mode+' Length of the dataset to traverse',length_data_set)
        return length_data_set
    

    def __getitem__(self,idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #Fetch Anchor Sample details
        sample = self.data_set.loc[idx]
        sample_img = cv2.imread(self.path_to_frames+sample.folder+'/'+sample.file+'.JPEG')
        sample_bbox = torch.tensor([sample.xmin,sample.ymin,sample.xmax,sample.ymax],dtype=torch.float).view(1,-1)

        sample_class = torch.tensor([sample.cat_code-1],dtype=torch.float).view(-1,1)

        other_sample_boxes,other_sample_classes = self.getOtherObjects(sample,self.data_set,self.all_data)
        if len(other_sample_boxes) > 0:
            sample_bbox = torch.cat([sample_bbox,other_sample_boxes],dim=0)
            sample_class = torch.cat([sample_class,other_sample_classes], dim=0)

    
        #Positive
        positive_candidate = self.data_set[
            (self.data_set.cat_code==sample.cat_code)& (self.data_set.snip_id==sample.snip_id) & (self.data_set.trackid==sample.trackid)]
            #CHECK! You might not need the same snip_id, you can just do same cat_code and same trackid --
        #dropping anchor or sample from positive candidates
        if len(positive_candidate) >= 2:
            positive_candidate = positive_candidate.drop(idx) #idx == sample.index
            positive = positive_candidate.loc[positive_candidate.index[0]]
        else:
            positive = sample

        #fetch positive sample details
        positive_img = cv2.imread(self.path_to_frames+positive.folder+'/'+positive.file+'.JPEG')
        positive_bbox = torch.tensor([positive.xmin,positive.ymin,positive.xmax,positive.ymax],dtype=torch.float).view(1,-1)
        positive_class = torch.tensor([positive.cat_code-1],dtype=torch.float).view(-1,1)

        #fetching other boxes
        other_positive_boxes,other_positive_classes = self.getOtherObjects(positive,self.data_set,self.all_data)
        if len(other_positive_boxes) > 0:
            positive_bbox = torch.cat([positive_bbox,other_positive_boxes],dim=0)
            positive_class = torch.cat([positive_class,other_positive_classes], dim=0)
        
        
        #Negative
        #WE WONT KEEP THE SNIP_ID==sample.SNIP_ID BECAUSE THE SAME IMAGE WITH 2 DIFFERENT OBJECTS COULD BE CONSIDERED NEGATIVE
        negative_candidate = self.data_set[
            (self.data_set.cat_code==sample.cat_code)& (self.data_set.snip_id!=sample.snip_id) &(self.data_set.trackid==sample.trackid)]
    
        if len(negative_candidate) == 0:
            negative_candidate = self.data_set[(self.data_set.cat_code!=sample.cat_code)& (self.data_set.snip_id!=sample.snip_id)]

        negative_candidate = negative_candidate.sample(n=1)
        negative = negative_candidate.loc[negative_candidate.index[0]]

        #fetch negative sample details
        negative_img = cv2.imread(self.path_to_frames+negative.folder+'/'+negative.file+'.JPEG')
        negative_bbox = torch.tensor([negative.xmin,negative.ymin,negative.xmax,negative.ymax],dtype=torch.float).view(1,-1)
        negative_class = torch.tensor([negative.cat_code-1],dtype=torch.float).view(-1,1)

        #fetching other boxes
        other_negative_boxes,other_negative_classes = self.getOtherObjects(negative,self.data_set,self.all_data)
        if len(other_negative_boxes) > 0:
            negative_bbox = torch.cat([negative_bbox,other_negative_boxes],dim=0)
            negative_class = torch.cat([negative_class,other_negative_classes], dim=0)


        #Augmentations
       

        if self.split_mode == 'train':
            
            #Box Augmentations
            sample_img, sample_bbox = self.aug.random_flip(sample_img, sample_bbox)
            positive_img, positive_bbox = self.aug.random_flip(positive_img, positive_bbox)
            negative_img, negative_bbox = self.aug.random_flip(negative_img, negative_bbox)
            
            sample_img, sample_bbox = self.aug.random_scale(sample_img, sample_bbox)
            positive_img, positive_bbox = self.aug.random_scale(positive_img, positive_bbox)
            negative_img, negative_bbox = self.aug.random_scale(negative_img, negative_bbox)
            
            #Non-box augmentations
            sample_img = self.aug.random_blur(sample_img)
            positive_img = self.aug.random_blur(positive_img)
            negative_img = self.aug.random_blur(negative_img)

            '''sample_img = self.aug.random_brightness(sample_img)
            positive_img = self.aug.random_brightness(positive_img)
            negative_img = self.aug.random_brightness(negative_img)'''

            sample_img = self.aug.random_hue(sample_img)
            positive_img = self.aug.random_hue(positive_img)
            negative_img = self.aug.random_hue(negative_img)

            sample_img = self.aug.random_saturation(sample_img)
            positive_img = self.aug.random_saturation(positive_img)
            negative_img = self.aug.random_saturation(negative_img)

            sample_img, sample_bbox, sample_class = self.aug.random_shift(sample_img, sample_bbox, sample_class)
            positive_img, positive_bbox, positive_class = self.aug.random_shift(positive_img, positive_bbox, positive_class)
            negative_img, negative_bbox, negative_class = self.aug.random_shift(negative_img, negative_bbox, negative_class)

            ''' sample_img, sample_bbox, sample_class = self.aug.random_crop(sample_img, sample_bbox, sample_class)
            positive_img, positive_bbox, positive_class = self.aug.random_crop(positive_img, positive_bbox, positive_class)
            negative_img, negative_bbox, negative_class = self.aug.random_crop(negative_img, negative_bbox, negative_class)'''

        #Division issue

        h_sample, w_sample, _ = sample_img.shape
        h_positive, w_positive, _ = positive_img.shape
        h_negative, w_negative, _ = negative_img.shape

        #Normalize Bounding boxes

        sample_bbox = sample_bbox/((torch.tensor([w_sample,h_sample,w_sample,h_sample],dtype=torch.float)).view(1,-1))

        positive_bbox = positive_bbox/((torch.tensor([w_positive,h_positive,w_positive,h_positive],dtype=torch.float)).view(1,-1))

        negative_bbox = negative_bbox/((torch.tensor([w_negative,h_negative,w_negative,h_negative],dtype=torch.float)).view(1,-1))


        #Encoding Target
        sample_target = self.encodeTarget_locem(sample_bbox,sample_class,1)

        positive_target = self.encodeTarget_locem(positive_bbox,positive_class,2)

        negative_target = self.encodeTarget_locem(negative_bbox,negative_class,3)


        #Image Resize    

        sample_img = cv2.resize(sample_img, dsize=(self.network_dim, self.network_dim), interpolation=cv2.INTER_LINEAR)

        positive_img = cv2.resize(positive_img, dsize=(self.network_dim, self.network_dim), interpolation=cv2.INTER_LINEAR)

        negative_img = cv2.resize(negative_img, dsize=(self.network_dim, self.network_dim), interpolation=cv2.INTER_LINEAR)

        #CVTCOLOR
        #sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        #positive_img = cv2.cvtColor(positive_img, cv2.COLOR_BGR2RGB)
        #negative_img = cv2.cvtColor(negative_img, cv2.COLOR_BGR2RGB)

        #Image Normalization - # normalize from -1.0 to 1.0.

        sample_img = (sample_img - self.mean) / 255.0
        positive_img = (positive_img - self.mean) / 255.0
        negative_img = (negative_img - self.mean) / 255.0
        
        #Image to Tensor
        sample_img = self.to_tensor(sample_img)
        positive_img = self.to_tensor(positive_img)
        negative_img = self.to_tensor(negative_img)

        #Stacking images and tensors 

        images = torch.stack([sample_img,positive_img,negative_img],dim=0)
        target = torch.stack([sample_target,positive_target,negative_target], dim=0)

        
        #images = np.stack([sample_img,positive_img,negative_img],axis=0)
        #target = torch.stack([sample_bbox,positive_bbox,negative_bbox],dim=0)

        #return sample_img,positive_img,negative_img,sample_bbox,positive_bbox,negative_bbox
        return images,target
            
