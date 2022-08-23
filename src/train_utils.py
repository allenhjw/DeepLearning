# _train_utils.
# Function call to train any DL network using pytorch.

# Dedepencies

import torch.nn as nn
import argparse as ap
import torch.optim as opt
from torch.utils.data import DataLoader
import joblib
import logging as lg
import time
import torch as tr
tr.backends.mps.is_available()

for handler in lg.root.handlers[:]:
    lg.root.removeHandler(handler)
# configure log file
lg.basicConfig(filename='/Users/allenhum/Documents/work/code/model_train.log',filemode='a',level=lg.INFO,
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


class _trainDL(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__()
        # check if device avaliable. 
        if device != 'cpu':
            if tr.backends.mps.is_available():
                self.run_device=device
            else:
                 self.run_device='cpu'
        self.run_device=device

    
    def _passAgrTrg(self,INIT_lt=0.01,Epoch=10,lr_gama=0.1):
        '''
            Call funtion use to create the parser for training parameters.\n

            Input arg:\n
            ----------\n
            INIT_lt: (dtype: float): setup the initial learning rate for the network. Default is 0.01.\n
            Epoch: (dtype: int): setup the initial training iteration, i.e., Epoch. Default is 10.\n

            Return arg:\n
            ----------\n
            param RETURN: vars: (dtype: parser): store the training parameters.\n
        '''
        parser=ap.ArgumentParser()
        parser.add_argument('-e','--epochs',default=Epoch,type=int,help='Number of training epochs.')
        parser.add_argument('-l','--learning rate',default=INIT_lt,type=float,help='the learning rate for the network.')
        parser.add_argument('-g','--learning rate gama',default=lr_gama,type=float,help='learning rate scheduler.')
        lg.info('training parameters parse')
        lg.info('epochs:{}'.format(Epoch))
        lg.info('learning rate:{}'.format(INIT_lt))
        lg.info('learning rate gama:{}'.format(lr_gama))

        return vars(parser.parse_args()) 
    
    def _trainNN(self,mdl,arg,in_trgdata):
        '''
            Function to train the network using split data\n
            
            Input arg:\n
            ----------\n
            mdl (dtype: pytorch model):  input model.\n
            arg (dtype: Argument Parser): parameters to train the model.\n
            in_trgdata (dtype: data tensor): training data in tensor.\n

            Return arg:\n
            ----------\n
            mdl (dtype: pytorch model):  trained model.\n

        '''
        in_trgdata=DataLoader(in_trgdata,batch_size=64,shuffle=True) #pin_memory_device='mps:0',pin_memory=True
        #in_trgdata=in_trgdata(self.run_device)
        optim=opt.Adam(mdl.parameters(),lr=arg['learning rate'],amsgrad=True,weight_decay=1e-6,betas=[0.5,0.999])
        lr_schder=opt.lr_scheduler.ExponentialLR(optim,gamma=0.1)
        loss_func=nn.CrossEntropyLoss()
        mdl.to(self.run_device)
        mdl.train()
        for X_ep in range (0,arg['epochs']):
            sec_time=time.time()
            counter=0
            acc_loss=0
            for x,y in in_trgdata:
                # get the inputs; data is a list of [inputs, labels]
                optim.zero_grad()
                x=x.to(self.run_device)
                NN_out=mdl._forward(x)
                y=y.to(self.run_device)
                loss=loss_func(NN_out,y)
                loss.backward()
                optim.step()
                # log the loss
                acc_loss+=loss.item()
                counter+=1
            lr_schder.step()
            print('Seconds since epoch =',sec_time)
            print(f'Loss at epoch {X_ep}: {acc_loss/counter}')
            lg.info('Loss per epoch {} :{}'.format(X_ep,acc_loss/counter))
        return mdl

    def _saveTrgMdl(self,model,save_dir,filename):
        '''
            Function to save the model to user define directory\n
            
            Input arg:\n
            ----------\n
            model (dtype: pytorch model):  input model.\n
            save_dir (dtype:  str): saving directory.\n
            filename (dtype: str): filename.\n

            Return arg:\n
            ----------\n
            mdl (dtype: pytorch model):  trained model.\n

        '''
        joblib.dump(model,save_dir+filename)



