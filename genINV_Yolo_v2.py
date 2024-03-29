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
from augmentations import Augment

#gitcheck0

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
def rescaleBoundingBox(height,width,rescaled_dim,xmin,ymin,xmax,ymax):
    
    #Required CNN input dimensions are generally squares hence just one dimension, rescaled_dim
    scale_x = rescaled_dim/width
    scale_y = rescaled_dim/height

    xmax = int(xmax * scale_x)
    xmin = int(xmin * scale_x)
    ymax = int(ymax * scale_y)
    ymin = int(ymin * scale_y)
    
    return [xmin,ymin,xmax,ymax]
    

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
        
        print('USING v101 of generator')
    
        self.transform = transform
        self.root_datasets = root_datasets
        self.root_ImageNetVidsDevkit = self.root_datasets+"ImageNetVids/imageNetVidsDevkit.data/"
        self.root_ImageNetVids = self.root_datasets+"ImageNetVids/imageNetVids.data/"
        self.split_mode = split
        self.aug = Augment()


        if self.split_mode == 'train':
            self.path_to_frames=self.root_ImageNetVids+"Data/VID/train/"
            
        elif self.split_mode == 'val':
            self.path_to_frames=self.root_ImageNetVids+"Data/VID/val/"
           
        else:
            raise ValueError('Split has to be train or val')

        self.data_set = pd.read_pickle(path_to_dataset)
        self.unique_keys = self.getKeys(pd.DataFrame(self.data_set))

        self.network_dim = 224
        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

        self.to_tensor = transforms.ToTensor()

        self.S,self.B,self.C,self.X,self.beta,self.gamma = 7,2,30,5,64,1

    def encodeTarget_locem(self,target_bbox,target_class,target_id):
        '''
            target_bbox = [xmax,xmin,ymax,ymin] #~~ [x2,x1,y2,y1]
            target_class = int
            target_id = int
            target_output = (S,S,B*X+C+gama)

        '''
        #Figure out label encoding

        S = 7
        B = 2
        X = 5
        C = 30
        #gamma = 1

        if target_class.size(0) > 1:
            raise ValueError('Need to adjust for target_class > 1')

        labels = target_class

        #v1
        #boxes = torch.Tensor(target_bbox).view(1,-1)
        #v101
        boxes = target_bbox
        #print('pre-norm',boxes)
        

        image_height,image_width = 224,224
        #boxes /= torch.Tensor([[image_width, image_height, image_width, image_height]]).expand_as(boxes) # normalize (x1, y1, x2, y2) w.r.t. image width/height.
        #print('post-norm',boxes)
        #Figure out label encoding
        #labels = [target_class] #type(target_class)=list(int) 

        target = torch.zeros(S, S, B*X + C )
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
                s = 5 * k
                target[j, i, s  :s+2] = xy_normalized
                target[j, i, s+2:s+4] = wh
                target[j, i, s+4    ] = 1.0 #Confidence of the bounding box
            
            #Setting class label
            target[j, i, B*X + label] = 1.0

            #Setting gamma label
            

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
        
        #Fetch Anchor Sample details
        sample = self.data_set.loc[idx]
        sample_img = cv2.imread(self.path_to_frames+sample.folder+'/'+sample.file+'.JPEG')
        sample_bbox = torch.tensor([sample.xmin,sample.ymin,sample.xmax,sample.ymax],dtype=torch.float).view(1,-1)

        #print('PRE_NORM',sample_bbox)

        sample_class = torch.tensor([sample.cat_code-1]).view(-1,1)


        #Augmentations
       

        if self.split_mode == 'train':
            

            #Box Augmentations
            sample_img, sample_bbox = self.aug.random_flip(sample_img, torch.Tensor(sample_bbox))
            
            sample_img, sample_bbox = self.aug.random_scale(sample_img, sample_bbox)
            
            #Non-box augmentations
            sample_img = self.aug.random_blur(sample_img)
            sample_img = self.aug.random_brightness(sample_img)
            sample_img = self.aug.random_hue(sample_img)
            sample_img = self.aug.random_saturation(sample_img)
            
            sample_img, sample_bbox, sample_class = self.aug.random_shift(sample_img, sample_bbox, sample_class)
        
            sample_img, sample_bbox, sample_class = self.aug.random_crop(sample_img, sample_bbox, sample_class)
           

        #Division issue

        h_sample, w_sample, _ = sample_img.shape
 
        #Normalize Bounding boxes

        sample_bbox = sample_bbox/(torch.tensor([w_sample,h_sample,w_sample,h_sample],dtype=torch.float)).view(1,-1)

        #Encoding Target
        sample_target = self.encodeTarget_locem(sample_bbox,sample_class,1)

        #Image Resize    

        sample_img = cv2.resize(sample_img, dsize=(self.network_dim, self.network_dim), interpolation=cv2.INTER_LINEAR)

        #CVTCOLOR
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

        #Image Normalization - # normalize from -1.0 to 1.0.

        sample_img = (sample_img - self.mean) / 255.0


        #Image to Tensor
        sample_img = self.to_tensor(sample_img)


        #Stacking images and tensors 
        images = sample_img
        target = sample_target

        '''images = torch.stack([sample_img,positive_img,negative_img],dim=0)
        target = torch.stack([sample_target,positive_target,negative_target], dim=0)
        '''

        return images,target

    def check_box(self, box, width, height):
        
        for i in range(box.size(1)):
            if i % 2 ==0: #x1,x2
                if box[0][i] > width:
                    print('Problemo -x',box[0][i])
            else:
                if box[0][i] > height:
                    print('Problemo -y',box[0][i])
            
