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

def xxyy_to_xywh(x0,y0,x1,y1):
    
    width = x1-x0
    height = y1-y0
    xc = x0+(width/2)
    yc = y0+(height/2)
    
    return xc,yc,width,height
def xywh_to_xxyy(xc,yc,width,height):
    
    y0 = yc-(height/2)
    x0 = xc-(width/2)
    x1 = width+x0
    y1 = height+y0
    
    return [(x0,y0),(x1,y1)]
    

def cropImage(sample,image,xmax,xmin,ymax,ymin):

    '''if not isinstance(image, np.ndarray):
            raise ValueError("img is not numpy array -crop",sample)'''
    
    #img = image[ymin:ymax,xmin:xmax,:]
    #img = cv2.resize(image,dsize=(224,224), interpolation = cv2.INTER_AREA)

    img = image.crop((xmin,ymin,xmax,ymax))
    img = img.resize((224,224))
    
    return img
def rescaleBoundingBox(height,width,rescaled_dim,xmax,xmin,ymax,ymin):
    
    #Required CNN input dimensions are generally squares hence just one dimension, rescaled_dim
    scale_x = rescaled_dim/width
    scale_y = rescaled_dim/height

    xmax = int(xmax * scale_x)
    xmin = int(xmin * scale_x)
    ymax = int(ymax * scale_y)
    ymin = int(ymin * scale_y)
    
    return [xmax,xmin,ymax,ymin]
    

def drawBoundingBox(self,img,xmax,xmin,ymax,ymin):
  
      image = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),2)
      #get() converts cv2.umat to ndarray
      return plt.imshow(image.get())

def drawBoundingBox_xxyy(img,x1,y1,x2,y2):

    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([(x1,y1),(x2,y2)],outline=(255,255,255,0),width=5)
    return img

class ImageNetVID(data.Dataset):


    def __init__(self,root_datasets,path_to_dataset,split,transform=None):
        
    
        self.transform = transform
        self.root_datasets = root_datasets
        self.root_ImageNetVidsDevkit = self.root_datasets+"ImageNetVids/imageNetVidsDevkit.data/"
        self.root_ImageNetVids = self.root_datasets+"ImageNetVids/imageNetVids.data/"
        self.split_mode = split
        self.reduceCounter = 0


        if split == 'train':
            self.path_to_frames=self.root_ImageNetVids+"Data/VID/train/"
            
        elif split == 'val':
            self.path_to_frames=self.root_ImageNetVids+"Data/VID/val/"
           
        else:
            raise ValueError('Split has to be train or val')

        self.data_set = pd.read_pickle(path_to_dataset)
        self.unique_keys = self.getKeys(pd.DataFrame(self.data_set))

        self.network_dim = 224

        self.S,self.B,self.C,self.X,self.beta,self.gamma = 7,2,30,5,64,1

    def encodeTarget_locem(self,target_bbox,target_class,target_id):
        '''
            target_bbox = [xmax,xmin,ymax,ymin] #~~ [x2,x1,y2,y1]
            target_output = (S,S,B*X+C+gama)

        '''
        #Figure out label encoding

        S = 7
        B = 2
        X = 5
        C = 30
        gamma = 1

        x2,x1,y2,y1 = target_bbox
        target_bbox = [x1,y1,x2,y2]
        boxes = torch.Tensor([target_bbox])

        image_height,image_width = 224,224
        boxes /= torch.Tensor([[image_width, image_height, image_width, image_height]]).expand_as(boxes) # normalize (x1, y1, x2, y2) w.r.t. image width/height.
        #Figure out label encoding
        labels = [target_class] #type(target_class)=list(int) #CONSIDER! negation of +1 from encoder and -1 from main

        target = torch.zeros(S, S, B*X + C + gamma)
        cell_size = 1.0 / float(S)
        boxes_wh = boxes[:, 2:] - boxes[:, :2] # width and height for each box, [n, 2]
        boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0 # center x & y for each box, [n, 2]

        for b in range(boxes.size(0)):
            xy, wh, label = boxes_xy[b], boxes_wh[b], int(labels[b])

            ij = (xy / cell_size).ceil() - 1.0 
            i, j = int(ij[0]), int(ij[1]) # y & x index which represents its location on the grid.
            x0y0 = ij * cell_size # x & y of the cell left-top corner.
            xy_normalized = (xy - x0y0) / cell_size # x & y of the box on the cell, normalized from 0.0 to 1.0.

            for k in range(B):
                s = 5 * k
                target[j, i, s  :s+2] = xy_normalized
                target[j, i, s+2:s+4] = wh
                target[j, i, s+4    ] = 1.0 #Confidence of the bounding box
            
            #Setting class label
            target[j, i, B*X + label] = 1.0

            #Setting gamma label
            target[j, i, B*X + C] = target_id

            return target

    def class_decoder(self,pred_tensor,target_tensor,accuracy):

        '''
            Args:
            Out:
        '''

        S,B,C,X,beta,gamma = self.S,self.B,self.C,self.X,self.beta,self.gamma

        #Extract the mask of targets with a gamma value 1,2,3. This will give us a location as to where the boxes are and their class embedding respectively
        #gamma should be at 40?
        class_mask_target = (target_tensor[:,:,:,B*X+C]==1) | (target_tensor[:,:,:,B*X+C]==2) | (target_tensor[:,:,:,B*X+C]==3)

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

    

    '''def encodeTarget(self,target_bbox,target_class,target_id):
        
            #target_bbox = [xmax,xmin,ymax,ymin] #~~ [x2,x1,y2,y1]
            #target_output = (S,S,B*X+alpha+gama)

        
        import math

        #Realiging the bbox values
        x2,x1,y2,y1 = target_bbox
        target_bbox = [x1,y1,x2,y2]
        S = 7
        B = 2
        X = 5
        alpha = 30
        gamma = 1

        target_output = torch.zeros((S,S,B*X + alpha + gamma))
        w = x2-x1   #width of bounding box
        h = y2-y1   #height of bounding box
        xc,yc = (x1+x2)/2,(y1+y2)/2
        grid_size = 1./S

        bx,by = math.ceil(xc*grid_size)-1,math.ceil(yc*grid_size)-1

        #Placing bbox with at the correct location with a condfidence of 1
        target_output[bx,by,4]=1
        target_output[bx,by,9]=1 #second bbox
        #Setting class of sample
        target_output[bx,by,target_class.item()+9]=1

        #Setting gamma value of sample
        target_output[bx,by,B*X + alpha+gamma] = target_id

        #Setting xc,yc
        #Shifting origin to grid-cell
        grid_length = 224*grid_size
        xn,yn = bx*grid_length,by*grid_length
        xc,yc = xc-xn,yc-yn
        target_output[bx,by,:2]=xc
        target_output[bx,by,5:7]=yc

        #Setting w,h
        target_output[bx,by,2:4]=w
        target_output[bx,by,7:9]=h

        return target_output
    '''


    

    '''def decoderPrediction(self,pred_tensor):
        
        
            #pred_tensor = (batch_size,S,S,B*X+C+beta)

        
        S = 7
        B = 2
        X = 5
        C = 30
        gamma = 1
        beta = 64
        batch_size = pred_tensor.size(0)
        confidence_threshold = 0.5
        grid_size = 1./S
        grid_length = 224*grid_size
        

        pred_tensor.detach()

        out_dict = {}
        for b in batch_size:
            for s in S:
                for s2 in S:

                    c1 = pred_tensor[b,s,s2,4]
                    c2 = pred_tensor[b,s,s2,9]
                    argmax_conf = torch.argmax(torch.tensor((c1,c2))) #Needs to be replaced with conf >= threshold

                    class_pred = pred_tensor[b,s,s2,B*X:C]
                    #class_pred = torch.argmax(class_pred).item()+1
                    embed_pred = pred_tensor[b,s,s2,B*X+C:]

                    if argmax_conf == 0:
                        box = pred_tensor[b,s,s2,:4]
                        conf = c1
                    else:
                        box = pred_tensor[b,s,s2,5:9]
                        conf = c2

                    bx,by = s,s2
                    xn,yn = bx*grid_length,by*grid_length   
                    
                    box = [i.item() for i in box]
                    box[0]=box[0]+xn # Converting from grid-cell coord to image coord
                    box[1]=box[1]+yn #Same as above
                    out_dict[b]['box']=box
                    #CHECK! We need the top boxes
                    out_dict[b]['conf']=conf
                    out_dict[b]['class_pred']=class_pred
                    out_dict[b]['embedding']=embed_pred
        return out_dict
    '''



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
        
    def __len__(self):

        length_data_set = len(self.data_set)
        print(self.split_mode+' Length of the dataset to traverse',length_data_set)
        return length_data_set
    

    def __getitem__(self,idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.data_set.loc[idx]
        sample_img = Image.open(self.path_to_frames+sample.folder+'/'+sample.file+'.JPEG').resize((224,224))
        #sample_img = cropImage(sample,sample_img,sample.xmax,sample.xmin,sample.ymax,sample.ymin)
        sample_bbox = rescaleBoundingBox(sample.height,sample.width,self.network_dim,sample.xmax,sample.xmin,sample.ymax,sample.ymin)
        #sample_bbox = [sample.xmax,sample.xmin,sample.ymax,sample.ymin]
        sample_class = torch.tensor(sample.cat_code-1)

        sample_target = self.encodeTarget_locem(sample_bbox,sample_class,1)

        '''sample_id = self.unique_keys[
            (self.unique_keys.cat_code==sample.cat_code) & (self.unique_keys.snip_id==sample.snip_id) & (self.unique_keys.trackid==sample.trackid)]
        sample_id = torch.tensor(int(sample_id.index.values))'''

        #Positive
        positive_candidate = self.data_set[
            (self.data_set.cat_code==sample.cat_code)& (self.data_set.snip_id==sample.snip_id) & (self.data_set.trackid==sample.trackid)]
        #dropping anchor or sample from positive candidates
        if len(positive_candidate) >= 2:
            positive_candidate = positive_candidate.drop(idx) #idx == sample.index
            positive = positive_candidate.loc[positive_candidate.index[0]]
        else:
            positive = sample
        #fetch positive sample details
        positive_img = Image.open(self.path_to_frames+positive.folder+'/'+positive.file+'.JPEG').resize((224,224))
        #positive_img = cropImage(positive,positive_img,positive.xmax,positive.xmin,positive.ymax,positive.ymin)
        positive_bbox = rescaleBoundingBox(positive.height,positive.width,self.network_dim,positive.xmax,positive.xmin,positive.ymax,positive.ymin)
        #positive_bbox = [positive.xmax,positive.xmin,positive.ymax,positive.ymin]
        positive_class = torch.tensor(positive.cat_code-1)
        #positive_id = sample_id # Anchor and Positive have same trackIDS
        positive_target = self.encodeTarget_locem(positive_bbox,positive_class,2)

        #Negative
        #WE WONT KEEP THE SNIP_ID==sample.SNIP_ID BECAUSE THE SAME IMAGE WITH 2 DIFFERENT OBJECTS COULD BE CONSIDERED NEGATIVE
        negative_candidate = self.data_set[
            (self.data_set.cat_code==sample.cat_code)& (self.data_set.snip_id!=sample.snip_id) & (self.data_set.trackid!=sample.trackid)]

        if len(negative_candidate) == 0:
            negative_candidate = self.data_set[(self.data_set.cat_code==sample.cat_code)& (self.data_set.snip_id!=sample.snip_id) & (self.data_set.trackid!=sample.trackid)]
        
        if len(negative_candidate) == 0:
            negative_candidate = self.data_set[(self.data_set.cat_code!=sample.cat_code)& (self.data_set.snip_id!=sample.snip_id) & (self.data_set.trackid!=sample.trackid)]

        negative_candidate = negative_candidate.sample(n=1)
        negative = negative_candidate.loc[negative_candidate.index[0]]

        #fetch negative sample details
        negative_img = Image.open(self.path_to_frames+negative.folder+'/'+negative.file+'.JPEG').resize((224,224))
        #negative_img = cropImage(negative,negative_img,negative.xmax,negative.xmin,negative.ymax,negative.ymin)
        negative_bbox = rescaleBoundingBox(negative.height,negative.width,self.network_dim,negative.xmax,negative.xmin,negative.ymax,negative.ymin)
        #negative_bbox = [negative.xmax,negative.xmin,negative.ymax,negative.ymin]
        negative_class = torch.tensor(negative.cat_code-1)

        #fetch id
        '''negative_id = self.unique_keys[
            (self.unique_keys.cat_code==negative.cat_code) & (self.unique_keys.snip_id==negative.snip_id) & (self.unique_keys.trackid==negative.trackid)]
        negative_id = torch.tensor(int(negative_id.index.values))'''

        negative_target = self.encodeTarget_locem(negative_bbox,negative_class,3)
        

        if self.transform:
            #Transform label?
            sample_img = self.transform(sample_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        images = torch.stack([sample_img,positive_img,negative_img],dim=0)
        target = torch.stack([sample_target,positive_target,negative_target], dim=0)

        #output = {'input':images,'target':target}
        

        return images,target
