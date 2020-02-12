import pandas as pd
import numpy as np
import os, cv2
import matplotlib.pyplot as plt

class utils():

    def __init(self):
        #placeholder

    def showTorchTensorImg(self,img):
        
        #assert img is a torch of size nchw
        return plt.imshow(img.permute(1,2,0).numpy())

    def getImage(self,image_address):
        
        img = cv2.imread(image_address)
        #Interpolation not required
        img = cv2.resize(img,dsize=(224,224), interpolation = cv2.INTER_CUBIC)
    
        return img

    def drawBoundingBox(self,self,img,xmax,xmin,ymax,ymin):
    
        image = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),2);
        #get() converts cv2.umat to ndarray
        return plt.imshow(image.get())