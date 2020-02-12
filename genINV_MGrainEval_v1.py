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


def cropImage(sample,image,xmax,xmin,ymax,ymin):

    '''if not isinstance(image, np.ndarray):
            raise ValueError("img is not numpy array -crop",sample)'''
    
    #img = image[ymin:ymax,xmin:xmax,:]
    #img = cv2.resize(image,dsize=(224,224), interpolation = cv2.INTER_AREA)

    img = image.crop((xmin,ymin,xmax,ymax))
    img = img.resize((224,224))
    
    return img
    

def drawBoundingBox(self,img,xmax,xmin,ymax,ymin):
  
      image = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),2)
      #get() converts cv2.umat to ndarray
      return plt.imshow(image.get())

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
        sample_img = Image.open(self.path_to_frames+sample.folder+'/'+sample.file+'.JPEG')
        sample_img = cropImage(sample,sample_img,sample.xmax,sample.xmin,sample.ymax,sample.ymin)
        #sample_class = torch.tensor(sample.cat_code-1)
        sample_class = sample.cat_code-1

        sample_id = self.unique_keys[(self.unique_keys.cat_code==sample.cat_code) & (self.unique_keys.snip_id==sample.snip_id) & (self.unique_keys.trackid==sample.trackid)]
        #sample_id = torch.tensor(int(sample_id.index.values))
        
        sample_id = int(sample_id.index.values)


        if self.transform:
            #Transform label?
            sample_img = self.transform(sample_img)

        output = {'input':sample_img,'classifier_target':sample_class,'instance_target':sample_id}

        return output
        
       
        
        '''unique_id = self.unique_ids.loc[idx]

        ds_idx = self.data_set[(self.data_set.cat_code == unique_id.cat_code) & (self.data_set.snip_id == unique_id.snip_id) & (self.data_set.trackid == unique_id.trackid)]

        ds_idx = ds_idx.sample(n=1)
        trip_index = [index_data_set for index_data_set in ds_idx.index]

        sample = ds_idx.loc[trip_index[0]]
        sample_img = Image.open(self.path_to_frames+sample.folder+'/'+sample.file+'.JPEG')
        sample_img = cropImage(sample,sample_img,sample.xmax,sample.xmin,sample.ymax,sample.ymin)
        sample_class = torch.tensor(sample.cat_code-1)

        #print('TYPE SAMPLE_IMG',type(sample_img))


        if self.transform:
                #sample = self.transform(sample)
                sample_img = self.transform(sample_img)

            
        return [sample_img,sample_class]'''

