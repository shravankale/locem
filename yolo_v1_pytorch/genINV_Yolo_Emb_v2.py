import pandas as pd
import numpy as np
import os, fnmatch, re, cv2, random,sys
import torch
sys.path.append('..')

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torch.utils import data
from torchvision import transforms
from PIL import Image
from util.augmentations import Augment

class ImageNetVID_RTR(data.Dataset):


    def __init__(self,root_datasets,path_to_dataset,split,image_size,transform=None):
        
        print('USING v101 of generator')
    
        self.transform = transform
        self.root_datasets = root_datasets
        #self.root_ImageNetVidsDevkit = self.root_datasets+"ImageNetVids/imageNetVidsDevkit.data/"
        #self.root_ImageNetVids = self.root_datasets+"ImageNetVids/imageNetVids.data/"
        self.split_mode = split
        


        if self.split_mode == 'train':
            self.path_to_frames=self.root_datasets+"ILSVRC2015/Data/VID/train/"
            self.aug = Augment()
            
        elif self.split_mode == 'val':
            self.path_to_frames=self.root_datasets+"ILSVRC2015/Data/VID/val/"
           
        else:
            raise ValueError('Split has to be train or val')
        
        self.map_vid = pd.read_pickle("../data/map_vid.pkl")
        self.map_cat = self.map_vid.to_dict()['category_name']

        self.data_set = pd.read_pickle(path_to_dataset)
        self.unique_keys = self.getKeys(pd.DataFrame(self.data_set))

        self.network_dim = image_size
        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        #mean_rgb = [0.485, 0.456, 0.406]
        self.mean = np.array(mean_rgb, dtype=np.float32)
        #std_values = [255,255,255]
        std_values = [0.229, 0.224, 0.225]
        self.std = np.array(std_values, dtype=np.float32)

        self.to_tensor = transforms.ToTensor()

        #self.S,self.B,self.C,self.X,self.beta,self.gamma = 7,2,30,5,64,1


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

    def preprocessInputImage(self,img,network_dim,mean):

        img = cv2.resize(img, dsize=(network_dim, network_dim), interpolation=cv2.INTER_LINEAR)
        #img = img/255 #Converts RGB(0,255) to RGB(0,1)
        #img = (img - self.mean) / self.std #Converting to range [-1,1]
        img = (img - mean)/255.0

        return img

    def unnormalizeImage(self,img):

        img = (img *self.std) + mean
        img = img*255 #Converting RGB(0,1) to RGB(0,255)

        return img

        
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
        sample_category = self.map_cat[sample.cat_code-1]
        sample_class = torch.tensor([sample.cat_code-1]).view(-1,1)

        '''if self.split_mode == 'train':
            

            #Box Augmentations
            sample_img, sample_bbox = self.aug.random_flip(sample_img, torch.Tensor(sample_bbox))
            
            sample_img, sample_bbox = self.aug.random_scale(sample_img, sample_bbox)
           
            #Non-box augmentations
            sample_img = self.aug.random_blur(sample_img)            

            sample_img = self.aug.random_brightness(sample_img)

            sample_img = self.aug.random_hue(sample_img)

            sample_img = self.aug.random_saturation(sample_img)

            sample_img, sample_bbox, sample_class = self.aug.random_shift(sample_img, sample_bbox, sample_class)

            sample_img, sample_bbox, sample_class = self.aug.random_crop(sample_img, sample_bbox, sample_class)'''

        
        x1,y1,x2,y2 = sample_bbox.numpy()[0].astype(int)
        sample_img = sample_img[y1:y2,x1:x2,:]
        

        #Augmentations

        sample_img = self.preprocessInputImage(sample_img,self.network_dim,self.mean)
        sample_img = self.to_tensor(sample_img)
        sample_class = sample_class[0][0]


        ids = self.unique_keys[(self.unique_keys.cat_code==sample.cat_code) & (self.unique_keys.snip_id==sample.snip_id) & (self.unique_keys.trackid==sample.trackid)].index.to_numpy()
        ids = int(ids)

        return [sample_img,sample_class,ids]

        '''sample_bbox = torch.tensor([sample.xmin,sample.ymin,sample.xmax,sample.ymax],dtype=torch.float).view(1,-1)

        #print('PRE_NORM',sample_bbox)

        sample_class = torch.tensor([sample.cat_code-1]).view(-1,1)'''


        #Augmentations
       

        '''if self.split_mode == 'train':
            

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
        '''
        #Division issue

        '''h_sample, w_sample, _ = sample_img.shape

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
        sample_img = self.to_tensor(sample_img)'''

        #return [sample_img,idx]


            
