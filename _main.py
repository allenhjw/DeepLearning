# _main function 
# 
# Dependencies

from tkinter import font
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from torch import true_divide
from torchvision.datasets import CIFAR10,ImageNet
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
import torch.multiprocessing as mp
import torch as tr
#device=tr.device('mps')
import os
import sys
import ssl
import joblib
import time
import random

# Set random seed for reproducibility
manualSeed = random.randint(1, 10000) 
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
tr.manual_seed(manualSeed)

from src import model_utils, train_utils,infer_utils,_grad_CAM_Mod,misc_functions

ssl._create_default_https_context = ssl._create_unverified_context
import logging as lg
for handler in lg.root.handlers[:]:
    lg.root.removeHandler(handler)
# locate data folder.
sys.path.append('/Users/allenhum/Documents/work')
sys.path.append('/Users/allenhum/Documents/work/code')
sys.path.append('/Users/allenhum/Documents/work/code/src')

# configure log file
lg.basicConfig(filename='/Users/allenhum/Documents/work/code/run.log',filemode='a',level=lg.INFO,
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


class _main():
    def __init__(self):
        self.workdir=sys.path[len(sys.path)-3] # ensure the directory is correct.
        pass
    def _LoadData(self):
        def _splitData(trg_data,trg_ratio=0.5):
            '''
                Function to further split the training data into Training and validate using the user define ratio.\n
                
                Input arg:\n
                ----------\n
                trg_data (dtype: numpy): the input training data.\n
                trg_ratio (dtype: float) : the split ratio between training and valid data set. Default is 0.5.\n

                Return arg:\n
                -----------\n
                return trg_data, valid_Data,test_data (dtype: numpy) : the split taining and valid data set.\n
            '''
            #val_ratio=1-trg_ratio
            trgData_len=len(trg_data)*trg_ratio
            valData_len=len(trg_data)-trgData_len
            (trg_Data, valid_Data)=random_split(trg_data,[int(trgData_len),int(valData_len)],
                                              generator=tr.Generator().manual_seed(42))
            return trg_Data,valid_Data

        lg.info('Download image data to {}'.format(self.workdir+'/data/'))
        data=CIFAR10(root=self.workdir+'/data/train/',train=True,download=True,
                     transform=transforms.Compose([transforms.ToTensor()]))
        trg_data,val_data=_splitData(data,0.6)
        self.train_data=DataLoader(trg_data,batch_size=64,shuffle=True)
        self.valid_data=DataLoader(val_data,batch_size=64,shuffle=True)
        data=CIFAR10(root=self.workdir+'/data/test/',train=False,download=True,
                     transform=transforms.Compose([transforms.ToTensor()]))
        self.test_data=DataLoader(data,batch_size=64,shuffle=False)
        test_data=data
        lg.info('Image data downloaded.')
        return trg_data,val_data,test_data#self.train_data, self.valid_data, self.test_data

    def _ViewData(self,split_type='train',h=0,w=0):
        '''
        #   Function use to enumerate through the batch of data set and visualise using seaborn-image.\n
        #   *Input Arg:\n
        #   1. split_type (dtype: str): User select the data type to view. 
        #       E.g., 'train' - view training data set.
        #             'test' - view testing data set.
        #             'none' - view all data set.\n
        #   2. h (dtype: int): User defines the number of image to display in the y axis. default is 0\n
        #   3. w (dtype: int): User defines the number of image to display in the x axis. default is 0\n
        #              
        '''
 
        f,axarr = plt.subplots(h,w)
      
        if split_type == 'train':
            for index,img_x in enumerate (self.train_data):
                # step through every images in the batch
                if index == 10: # only save the 1st 10 batches of image data
                    break
                for col in range (0,w):
                    for row in range(0,h):
                        img_ind=w*col+row
                        img=transforms.ToPILImage()(img_x[0][img_ind]).convert("RGB")
                        
                        #img=img_x[0][img_ind].detach().numpy()
                        #plt.axis('off')
                        axarr[row,col].axis('off')
                        axarr[row,col].imshow(img)
                        #fig.add_subplot(row+1,col+1,img)
                #plt.show()
                plt.suptitle("{} Image illustration: Batch {}".format('CIFAR10',str(index)),fontsize=18)
                plt.savefig(self.workdir+'/plots/{}_img_batch{}.jpeg'.format('CIFAR10',str(index)))
        elif split_type == 'test':
            for index,img_x in enumerate (self.test_data):
                # step through every images in the batch
                if index == 10: # only save the 1st 10 batches of image data
                    break
                for col in range (0,w):
                    for row in range(0,h):
                        img_ind=w*col+row
                        img=transforms.ToPILImage()(img_x[0][img_ind]).convert("RGB")
                        
                        #img=img_x[0][img_ind].detach().numpy()
                        #plt.axis('off')
                        axarr[row,col].axis('off')
                        axarr[row,col].imshow(img)
                        #fig.add_subplot(row+1,col+1,img)
                #plt.show()
                plt.suptitle("{} Image illustration test: Batch {}".format('CIFAR10',str(index)),fontsize=18)
                plt.savefig(self.workdir+'/plots/{}_img_test_batch{}.jpeg'.format('CIFAR10',str(index)))
        lg.info('Batches image data saved: {}.'.format(self.workdir+'/plots/'))
        lg.info('image size and channel:{} , {}.'.format(str(img.size),img.mode))


if __name__=='__main__':

    '''
        Evalution on grad CAM
    '''
    Data=_main()
    trg_data, val_Data, test_data=Data._LoadData()
    model=joblib.load('/Users/allenhum/Documents/work/model/'+'mdl3CNNLayer_WgINIT_27jul.sav')
    model=model.to('cpu') # mount to local to infer
    gCAM=_grad_CAM_Mod._gradCAM_Mod(model,3) #18,12
    orig_img,CAM_img,label=gCAM._forwardGradHOOK(test_data)
    misc_functions.save_class_activation_images(orig_img,CAM_img,'test',label)

    '''
       Training and save the model.
    '''
    #Data=_main()
    #trg_data, val_Data, test_data=Data._LoadData()
    #Data._ViewData(split_type='test',h=8,w=8)
    #model=model_utils.CNN_net(3,64,10)
    #mdltrg=train_utils._trainDL(device='mps:0')
    #trg_param=mdltrg._passAgrTrg(1e-5,40)
    #model_utils.weightINIT(model)._weightINIT() # inital the weight - not use 
    #lg.info('Weight initialised') # inital the weight - not use 
    #trg_mdl=mdltrg._trainNN(model,trg_param,trg_data)
    ## save the trained model
    ##joblib.dump(trg_mdl,'/Users/allenhum/Documents/work/model/mdl_7jul.sav')
    #Save_PH='/Users/allenhum/Documents/work/model/'
    #file_name='mdl3CNNLayer_[convStride]29jul.sav'
    #mdltrg._saveTrgMdl(trg_mdl,Save_PH,file_name)
    #lg.info('Model saved at loction:{}'.format(Save_PH+file_name))

    '''
        Infer using the model
    '''
    # infer to test the model
    #mdlInfer=infer_utils._infer()
    #mdlInfer._ModelInfer(trg_mdl,val_Data)
    
    



