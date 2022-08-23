# _grad_CAM 
# Gradient Class Activiation Map
# Function uses to visualise activiation map
# we follow the paper "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" 
# by R. Selvaraju et al from 
# https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf

# import dependencies
from cv2 import trace
import numpy as np
import pandas as pd
import torch as tr
from torch.utils.data import DataLoader
import joblib
import random
from PIL import Image
import torch.nn.functional as f


# create class
class _gradCAM_Mod():
    def __init__(self,model,LOI):
        '''
            Function to initialise grad CAM.\n
            
            Input argument:\n
            ---------------\n
            model: (dtype: pytorch model) : input model for study the attention. Note: the model must be trained.\n
            LOI: (dtype: int) : layer of interest (LOI) to visualise the attention. Note: the LOI must before softmax or flatten\n

        '''

        self.model=model
        self.LOI=LOI
        pass

    def save_gradient(self, grad):
        '''
            Function to save gradient.\n
            
            Input argument:\n
            ---------------\n
            grad: (dtype: tensor): layer gradient.\n
    
        '''
        self.gradients = grad

    def _forwardGradHOOK(self, x):
        '''
            Function pass the input per step in the network; if current level == LOI, save the gradient.\n
            
            Input argument:\n
            ---------------\n
            x: (dtype: tensor) : input vector data (start or intermedient) to the model.\n
        '''
        
        #get_imgInd=random(0,63)
        data=DataLoader(x,batch_size=64,shuffle=False) # in x, there are len(data) batch
        len(data)
        # we random select the image from the dataloader via batch
        #get_dataBATCH=random.randint(0,len(data)-1)
        # user set the batch number
        get_dataBATCH = 100
        for indBATCH,(x,y) in enumerate (data):
            if indBATCH==get_dataBATCH:
                # random get the target number 
                #get_imgInd=random.randint(0,len(x)-1)
                # user set the image target number index
                get_imgInd = 7
                x=tr.unsqueeze(x[get_imgInd],0)
                y=y[get_imgInd].detach().numpy()
                break
        input_img=x    
        NN_out=None
        LOI_out=None
        # x is the select image and y is the label.
        for mod_ind, module in enumerate (list(self.model.children())):

            x=module(x)
            if int(mod_ind) == self.LOI:
               if  module.__str__().split()[0].split('(')[0]=='Conv2d':
                    x.register_hook(self.save_gradient)  
                    LOI_out=x
        NN_out=x # save the LOI output. 
        
        # Create label back propagating
        Target_out = tr.FloatTensor(1, NN_out.size()[-1]).zero_()
        Target_out[0][y] = 1
        # Carry on till classifier.
        self.model.zero_grad()
        # Backward pass with specified target
        NN_out.backward(gradient=Target_out, retain_graph=True) #gradient via backpropagation
        # Get gradients at the LOI
        _gradients = self.gradients.data.numpy()[0]
        # Get LOI outputs
        _LOI = LOI_out.data.numpy()[0]
        # Get weights from gradients at the LOI
        weights = np.mean(_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam (classifier activation matrix)
        cam = np.ones(_LOI.shape[1:], dtype=np.float32)
        # element multiplying each weight with its conv output and then, sum to get the CAM 
        
        for i, w in enumerate(weights):
            cam += w * _LOI[i, :, :] # equation (2) in the cited paper.
        cam=tr.tensor(cam)
        ReLU_CAM=f.relu(cam) # equation (2) in the cited paper.
        ReLU_CAM=ReLU_CAM.detach().numpy()
        Wt_cam = np.maximum(ReLU_CAM, 0)
        Wt_cam = (Wt_cam - np.min(Wt_cam)) / (np.max(Wt_cam) - np.min(Wt_cam))  # Normalize between 0-1
        Wt_cam = np.uint8(Wt_cam * 255)  # Scale between 0-255 to visualize
        Wt_cam = np.uint8(Image.fromarray(Wt_cam).resize((input_img.shape[2],
                       input_img.shape[3]), Image.ANTIALIAS))/255
        input_img=tr.squeeze(input_img,0)   
        input_img=tr.transpose(input_img,-1,0)
        input_img=input_img.detach().numpy()
        input_img=Image.fromarray(np.uint8(input_img*255))          
        return input_img,Wt_cam,y

def main():
    # load the model
    working_path='/Users/allenhum/Documents/work/'
    model=joblib.load(working_path+'model/mdl_13jul.sav')

    gCAM=_gradCAM_Mod(model,3)
    gCAM._forwardGradHOOK()
    pass
if __name__=='__main__':
    main()
    