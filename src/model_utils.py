# src utilites
# Objectives: utilies create basic structure of 2D deep learning network that can be ensemble.
#             The utilies contains a list of model as follow,
#             
#             Model_list{} = 'CnnVAE','Cnn_Net','ResNET18','Faster-RCNN'


from modulefinder import Module
from tokenize import Exponent
from sklearn.utils import shuffle
from torch.utils.data import Dataset,DataLoader
import torch as tr
import torchvision as trv
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as Func # import the function of the layers
import numpy as np
import pandas as pd
import ssl
#from tensorflow.keras.datasets import cifar10
import logging
#from torchsummary import summary
import argparse as ap
import joblib
import logging as lg
import argparse as ap
import joblib
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots 
#import plotly.express as px
#import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200



class CnnVAE_01(nn.Module):
    '''
    Class to configure the CNN VAE network.\n
    '''
    def __init__(self,latent_size,in_size,in_channel):
        '''
            Initialise the CNN network.\n
            
            Input arg:\n
            ----------\n
            latent_size: (dtype: int) the size of the latent space.\n
            in_size: (dtype: float) the input size of the input data to the network.\n
            in_channel (dtype: int) the number of channel of the image.
        '''
        super(CnnVAE_01,self).__init__()
        self.latent_size=latent_size 
        self.in_size=in_size
        self.in_channel=in_channel

        self.encoder=nn.Sequential(
                        nn.Conv2d(in_channels=self.in_channel,out_channels=64,kernel_size=4,padding=1,stride=2), # output size is 14*14*64
                        nn.ReLU(),
                        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,padding=1,stride=2),# output size is 7*7*128
                        nn.ReLU(),
                        nn.Flatten(),# output size is 7*7*128 = 6272
                        nn.Linear(in_features=6272,out_features=1024),
                        nn.ReLU()
                        )
        
        self.fc1=nn.Linear(in_features=1024,out_features=self.latent_size) # this is used to calculate mu.
        self.fc2=nn.Linear(in_features=1024,out_features=self.latent_size) # this is used to calculate sigma.

        self.decoder=nn.Sequential(
                     nn.Linear(in_features=self.latent_size,out_features=1024), # output will be 1 * 1024
                     nn.ReLU(),
                     nn.Linear(in_features=1024,out_features=6272),
                     nn.Unflatten(128,(7,7)), # output will become a 2D space with dimension batch, height, width.
                     nn.ReLU(),
                     nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,padding=1,stride=2),
                     nn.ReLU(),
                     nn.ConvTranspose2d(in_channels=64,out_channels=1,kernel_size=4,padding=1,stride=2),
                     nn.Sigmoid()
                     )
    
    def _forward(self,inputData):
        Encod_output=self.encoder(inputData)
        self.mu=self.fc1(Encod_output)
        self.sigma=self.fc2(Encod_output)
        # Calculate VAE re-parameterisation.
        self.repara = self.mu + self.sigma*tr.distributions.Normal(0, 1).sample(self.mu.shape) # Z value matrix
        # Calculate KL divergence.
        self.kl = (self.sigma**2 + self.mu**2 - tr.log(self.sigma) - 1/2).sum()
        pass

class CNN_net(nn.Module):
    def __init__(self,in_Channel=1,in_Batch=0,out_classes=0):
        '''
            Initialise the CNN network.\n
            
            Input arg:\n
            ----------\n
            in_Channel: (dtype: int) the numnber of channels. i.e., 1= Grayscale, 3=RBG. Default is set to 1.\n
            in_Batch: (dtype: int) input batch size data. Default is set to 0.\n
            out_classes (dtype: int) the number class to predict. Default is set to 0.\n
        '''
        super().__init__()
        # define CNN layer
        # initalise the 1st layer CNN: CNN2d->ReLU->Maxpooling
        # input image size: eg., Nx3x32x32; N= batch number
        self.conv1=nn.Conv2d(in_channels=in_Channel,out_channels=in_Batch,padding=0,kernel_size=5,stride=1) 
        # output size: eg., 16x16x64
        #self.relu1=nn.ReLU()
        self.BatchNorm1=nn.BatchNorm2d(in_Batch)
        self.Lrelu1=nn.LeakyReLU(0.2,inplace=True)
        #self.maxpool1=nn.MaxPool2d(padding=0,kernel_size=3,stride=1)
        
        # output size: eg., 26x26x64

        # initalise the 2nd layer CNN: CNN2d->ReLU->Maxpooling
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,
                             padding=0,kernel_size=5,stride=1)
        # output size: eg., 24x24x128                   
        self.BatchNorm2=nn.BatchNorm2d(128)
        self.Lrelu2=nn.LeakyReLU(0.2,inplace=True)
        #self.relu2=nn.ReLU()
        #self.maxpool2=nn.MaxPool2d(padding=0,kernel_size=3,stride=1)
        # output size: eg., 20x20x128

        # initalise the 3rd layer CNN: CNN2d->ReLU->Maxpooling
        self.conv3=nn.Conv2d(in_channels=128,out_channels=128,
                             padding=0,kernel_size=5,stride=1)
        # output size: eg., 16x16x256                   
        self.BatchNorm3=nn.BatchNorm2d(128)
        self.Lrelu3=nn.LeakyReLU(0.2,inplace=True)
        #self.relu3=nn.ReLU()
        #self.dpout3=nn.Dropout(0.8)
        #self.maxpool3=nn.MaxPool2d(padding=0,kernel_size=3,stride=1)
        # output size: eg., 14x14x256

        # initalise the 4th layer CNN: CNN2d->ReLU->Maxpooling
        self.conv4=nn.Conv2d(in_channels=128,out_channels=256,
                             padding=0,kernel_size=5,stride=1)
        #output size: eg., 10x10x512
        self.BatchNorm4=nn.BatchNorm2d(256)
        self.Lrelu4=nn.LeakyReLU(0.2,inplace=True)                  
        #self.relu4=nn.ReLU()
        #self.dpout4=nn.Dropout(0.8)
        #self.maxpool4=nn.MaxPool2d(padding=0,kernel_size=3,stride=1)
        # output size: eg., 8x8x512

        # initalise the 5th layer CNN: CNN2d->ReLU->Maxpooling
        self.conv5=nn.Conv2d(in_channels=256,out_channels=256,
                             padding=0,kernel_size=5,stride=1)
        self.BatchNorm5=nn.BatchNorm2d(256)
        self.Lrelu5=nn.LeakyReLU(0.2,inplace=True)
        # output size: eg., 4x4x1024                  
        #self.relu5=nn.ReLU()
        #self.dpout5=nn.Dropout(0.8)
        #self.maxpool5=nn.MaxPool2d(padding=0,kernel_size=3,stride=1)
        # output size: eg., 2x2x1024  

        # initalise the 6th layer CNN: CNN2d->ReLU->Maxpooling
        self.conv6=nn.Conv2d(in_channels=256,out_channels=512,
                             padding=0,kernel_size=5,stride=1)
        self.BatchNorm6=nn.BatchNorm2d(512)
        self.Lrelu6=nn.LeakyReLU(0.2,inplace=True)

        # initalise the 7th layer CNN: CNN2d->ReLU->Maxpooling
        self.conv7=nn.Conv2d(in_channels=512,out_channels=512,
                             padding=1,kernel_size=4,stride=2)
        self.BatchNorm7=nn.BatchNorm2d(512)
        self.Lrelu7=nn.LeakyReLU(0.2,inplace=True)

        # initialise flatten layer
        self.flat=nn.Flatten() # output size  2*2*128 = 512
        self.Fc1=nn.Linear(in_features=8192,out_features=8192)
        self.reluFc1=nn.ReLU()

        # initialise classifier layer
        self.Fc2 = nn.Linear(in_features=8192, out_features=out_classes)
        #self.relu4=nn.ReLU()
        #self.Fc3=nn.Linear(in_features=50,out_features=out_classes)
        self.dpoutFc=nn.Dropout(0.5)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    
    def _forward(self,in_Data):
        '''
            Function to feed the data into the define CNN.\n
            
            Input arg:\n
            ----------\n
            in_Data: (dtype: tensor) the tensor that store the input data.\n

            Return arg:\n
            -----------\n
            predict: (dtype: ) Class type.\n
        '''
        # Feed to the 1st CNN layer.
        x=self.conv1(in_Data)
        x=self.BatchNorm1(x)
        x=self.Lrelu1(x)

        # Feed to the 2nd CNN layer.
        x=self.conv2(x)
        x=self.BatchNorm2(x)
        x=self.Lrelu2(x)

        # Feed to the 3rd CNN layer.
        x=self.conv3(x)
        x=self.BatchNorm3(x)
        #x=self.dpout3(x)
        x=self.Lrelu3(x)

        # Feed to the 4th CNN layer.
        x=self.conv4(x)
        x=self.BatchNorm4(x)
        x=self.Lrelu4(x)
        #x=self.relu4(x)
        #x=self.dpout4(x)
        #x=self.maxpool4(x)

        # Feed to the 5th CNN layer.
        x=self.conv5(x)
        x=self.BatchNorm5(x)
        x=self.Lrelu5(x)
        #x=self.relu5(x)
        #x=self.dpout5(x)
        #x=self.maxpool5(x)

        # Feed to the 6th CNN layer.
        x=self.conv6(x)
        x=self.BatchNorm6(x)
        x=self.Lrelu6(x)

        # Feed to the 7th CNN layer.
        x=self.conv7(x)
        x=self.BatchNorm7(x)
        x=self.Lrelu7(x)

        # Feed to flatten layer and Fully connected layer (FC1).
        x=self.flat(x)
        x=self.Fc1(x)
        x=self.reluFc1(x)

        # Feed to classifier layer
        x=self.Fc2(x)
        #x=self.relu4(x)
        #x=self.Fc3(x)
        x=self.dpoutFc(x)
        output=self.logSoftmax(x)

        return output


    def _getLayerWeights(self,mdl):
        '''
            Function to get the weight of the convolution layers for visualise feature mapping.\n
            
            Input arg:\n
            ----------\n
            mdl (dtype: ) user define trained model. Important: the model must be pre-trained else the result is not going to be good.\n

            Return arg:\n
            -----------\n
            mdl_weights: (dtype: list) list contains the weight of the convolution layer.\n
            conv_lys: (dtype: list) list contains the convolution layer.\n
        '''
        # Save the conv layer weights in this list.
        mdl_weights =[]
        # Save the conv layers in this list.
        conv_lys = []
        # get all the model children as list
        mdl_child_ls = list(mdl.children())
        #counter to keep count of the conv layers
        counter = 0
        # append all the conv layers and their respective wights to the list
        
        for child in mdl_child_ls:
            if type(child)==nn.Conv2d:
                counter+=1
                mdl_weights.append(child.weight)
                conv_lys.append(child)
        print (f'Number of convolution layers: {counter}')

        return mdl_weights,conv_lys

class weightINIT():
    def __init__(self,model):
        self.model=model
        pass

    def _weightINIT(self):
        classname =self.model.__class__.__name__
        for X in self.model.children():
            if X.__class__.__name__ == 'Conv2d':
                nn.init.normal(X.weight.data,0,0.02) # set the mean to zero and STDDEV to 0.02 (Radford et al, ICLR 2016)
            elif X.__class__.__name__ == 'BatchNorm2d':
                nn.init.normal(X.weight.data,1,0.02) # set the mean to zero and STDDEV to 0.02 (Radford et al, ICLR 2016)
                nn.init.constant(X.bias.data,0) # set the mean to zero and STDDEV to 0.02 (Radford et al, ICLR 2016)
            pass
        #if classname.find('Conv') != -1:
        #    nn.init.normal_(self.model.weight.data, 0.0, 0.02)
        #elif classname.find('BatchNorm') != -1:
        #    nn.init.normal_(self.model.weight.data, 1.0, 0.02)
        #    nn.init.constant_(self.model.bias.data, 0)

class ResNET18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=0):
        '''
            Initialise the ResNET18 network.\n
            
            Input arg:\n
            ----------\n
            in_channels: (dtype: int) the numnber of channels. i.e., 1= Grayscale, 3=RBG. Default is set to 1.\n
            resblock: (dtype: class) initalise resblock class.\n
            outputs (dtype: int) the number class to predict. Default is set to 0.\n
        '''
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = tr.nn.AdaptiveAvgPool2d(1)
        self.fc = tr.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = tr.flatten(input)
        input = self.fc(input)

        return input

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)
    
class FasterRCNN(nn.Module):
    '''
        Class for creating Faster-RCNN. 
        
        Input arg:\n
        ----------\n
        BackModel: (dtype: Boolean) : if True, we will use a pre-train model, by default is ResNet50. 
        num_classes: (dtype: integer): the number of class to detect. Default is 2, i.e., Foregound and background.

    '''
    def __init__(self,BackModel=True,num_classes=2) -> None:
        super().__init__()

        import torchvision as trV
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        if BackModel: 
            # Load pre-trained model, resnet50 that is trained on COCO data. 
            self.BKmodel = trV.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            # get number of input features for the classifier
            in_features = self.BKmodel.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.BKmodel.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)





if __name__=='__main__':

    #a=CnnVAE_01(latent_size=2,in_size=64,in_channel=3)
    # using a.childern() to find the list of layer in the CNN
    #c=CNN_net(3,64,7)



    pass



