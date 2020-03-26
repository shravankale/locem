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
        

    def drawBoundingBox(self,img,xmin,ymin,xmax,ymax):
    
        image = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),2)
        #get() converts cv2.umat to ndarray
        return plt.imshow(image.get())

    def drawBoundingBox_xxyy(img,x1,y1,x2,y2):

        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([(x1,y1),(x2,y2)],outline=(255,255,255,0),width=5)
        return img