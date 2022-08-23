# Ultis to set up VAE encoder
# 

#
#from re import X
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
from tensorflow.keras.datasets import cifar10
import logging
from torchsummary import summary
import argparse as ap
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import plotly.express as px
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200



logging.basicConfig(filename='/Users/ahum/Documents/[Project] PIPE_DEV/src/scr_AEultis.log', 
                    encoding='utf-8',filemode='a',
                    format='<%(asctime)s> ---- %(message)s', level=logging.INFO)
#logging.info('Initialise cluster utilise')

class _Data():
    def __init__(self):
        pass
    def _GetData(self):
        '''
        Call function to download image data from user define source.\n

        Argument:\n
        ---------\n
        param RETURN: X_trg (dtype: np.array): stack images (response) split from the source for training.\n
        param RETURN: Y_trg (dtype: np.array): stack images split (predictor) from the source for training.\n 
        '''

        ssl._create_default_https_context = ssl._create_unverified_context
        #### load data and process
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        logging.info('Loading image data from {}'.format('cifar10'))
        return {'X_trg':x_train,'Y_trg':y_train}

    def _transform(self,in_data,type='feat'):
        '''
        Call function to transform the input data into tensor.

        Argument:\n
        ---------\n
        param IN: in_data (dtype:np.array): stack array of data (1d,2d or 3d data).\n
        param IN: type (dtype: string): 'feat' the stack array store the predictor or feature data. 'lab' the stack arry store the response 

        Return:\n
        -------\n
        param OUT: data_tensor (dtype: tensor): stack array of tensor data (1d,2d or 3d data).\n

        '''
        #Data_ls=[]
        if type =='feat':
            #data_tensor=tr.tensor(in_data)
            data_tensor=in_data/255
        # transform to floating btye
        #for ind,X in enumerate(in_data):
        #    X_b=trv.transforms.functional.to_tensor(X)
            #data_stack=tr.vstack(data_stack,data_tensor_b)
        #    Data_ls.append(X_b)
            return data_tensor
        #pass 


class _CLinearVAE(nn.Module):
    def __init__(self,in_IMG_arry,feat_space):
        '''
        Call function use to initise the VAE.\n

        Argument:\n
        ---------\n
        param IN: in_IMG_arry: (dtype: np.array): store the input image stack.\n
        param IN: feat_space: (dtype: integer): the size of the latent/feature space.\n
        '''
        super(_CLinearVAE, self).__init__()
        #in_IMG_arry.
        self.Get_IMG_SZ=in_IMG_arry.dataset.data.shape #X
        self.inData_SZ=self.Get_IMG_SZ[1]*self.Get_IMG_SZ[2]#*Get_IMG_SZ[3]
        self.feat_space=feat_space
        self.inData=in_IMG_arry
        self.AvgMAP_perBatch=[]
        self.AccAvgMAP=[]
        
        logging.info('Init linear network')
        pass

    def _NNconfig(self,C=1):
        '''
        Call function use to configure the structure of the network.\n

        Argument:\n
        ---------\n
        param IN: C: (dtype: integer): constant number use to define the size of the latent space

        '''
        # Set up the encoder layers using Pytorch;
        self.LayerEND1=nn.Linear(in_features=self.inData_SZ,out_features=512)
        self.LayerEND2=nn.Linear(in_features=512,out_features=self.feat_space*C)
        self.LayerEND3=nn.Linear(in_features=512,out_features=self.feat_space*C)
        
        #self.encoder=nn.Sequential(nn.Linear(in_features=self.inData_SZ,out_features=512),nn.ReLU()
                                    
        #                            ) #self.inData_SZ ;nn.Linear(in_features=512,out_features=self.feat_space*C),nn.ReLU(),nn.BatchNorm1d(num_features=self.feat_space*C)
        # set up the hidden layer using Pytorch;
        self.Latent=nn.Linear(in_features=512,out_features=self.feat_space*C)
        self.LatentmulogVar=nn.Linear(in_features=512,out_features=self.feat_space*C)

        # set up the decoder layer using Pytorch;
        self.LayerDEC1=nn.Linear(in_features=self.feat_space*C,out_features=512)
        self.LayerDEC2=nn.Linear(in_features=512,out_features=self.inData_SZ)
        
        #self.decoder=nn.Sequential(nn.Linear(in_features=self.feat_space*C,out_features=512),nn.ReLU(),
        #                            nn.Linear(in_features=512,out_features=self.inData_SZ),nn.Sigmoid()
        #                            )#self.inData_SZ
        logging.info('Encoder network configured')                           
        #pass
    

    def _forwardingENDCODER(self,x_data):
        # flatten the input data
        x_data_f=tr.flatten(x_data,start_dim=1)
        D_flow1=Func.relu(self.LayerEND1(x_data_f))
        D_flow2=self.LayerEND2(D_flow1) # there is no activation function to the latent space. 
        D_flow3=self.LayerEND3(D_flow1)
        #Calculate the mean of the latent space.
        mu=D_flow2
        sigma = tr.exp(D_flow3)
        repara = mu + sigma*tr.distributions.Normal(0, 1).sample(mu.shape) # Z value matrix
        #_a=Func.adaptive_avg_pool1d(repara,(-1,2)) # allow us to determine the global average for each channel. 
        kl = (sigma**2 + mu**2 - tr.log(sigma) - 1/2).sum()
        # For attention mapping purposes
        #attentionMAP=AttendMAP(D_flow1,repara)
        #a_k=attentionMAP._getScalar_a()
        #lat1,lat2=attentionMAP._getscaleMap(a_k['a_k'])
        #a_k_1=a_k['a_k'][:,1]
        #a_k_1_t=tr.tensor(a_k_1.reshape(D_flow1.shape[0],1))
        #self._attendMapDecode(lat1)
        return repara , kl, D_flow1

    def _forwardingDECODER(self,F_data):
        # Data flow from latent space to decoder.
        D_flow1 = Func.relu(self.LayerDEC1(F_data))
        D_flow2 = tr.sigmoid(self.LayerDEC2(D_flow1))
        return D_flow2.reshape((-1, 1, self.Get_IMG_SZ[1], self.Get_IMG_SZ[2]))

    def _attentionMAP(self,in_data,latent_data):
        attentionMAP=AttendMAP(in_data,latent_data)
        a_k=attentionMAP._getScalar_a()
        lat1,lat2=attentionMAP._getscaleMap(a_k['a_k'])
        #a_k_1=a_k['a_k'][:,1]
        #a_k_1_t=tr.tensor(a_k_1.reshape(D_flow1.shape[0],1))
        self._attendMapDecode(lat2)

    def _TrainStep2(self,mdl,arg,in_data):

        optim=opt.Adam(mdl.parameters(),lr=arg['learning rate'])
        lr_schder=opt.lr_scheduler.ExponentialLR(optim,gamma=0.1)
        for X_ep in range (0,arg['epochs']):
            for x,y in in_data:
                optim.zero_grad()
                repara, kl, p_layer =self._forwardingENDCODER(x)#in_data
                # get attention mapping
                #self._attentionMAP(p_layer,repara)
                ReCon_img=self._forwardingDECODER(repara)
                    #kl_loss = (-0.5*(1+NNparam['logVar'] - NNparam['mu']**2 - tr.exp(NNparam['logVar'])).sum(dim=1)).mean(dim=0)
                # calculate construction loss.
                Constr_loss=((x-ReCon_img)**2).sum()#*1/len(x)
                #kl_loss=0
                #recon_loss_criterion = nn.MSELoss()
                #recon_loss = recon_loss_criterion(x, NNparam['NN_out'].reshape(-1,x.shape[1],x.shape[2],x.shape[3]))
                NN_loss=Constr_loss+kl
                NN_loss.backward()
                optim.step()
                logging.info('loss per iteration:{}'.format(str(NN_loss)))
            lr_schder.step()
        return {'Model':mdl}

    def _attendMapDecode(self,in_MAP):
        # pass the attention map and decode
        #MapDEC1=nn.Linear(in_features=self.feat_space,out_features=512)
        def _plotIMG(in_data):
            '''
            Call function to plot the attention map

            Arg:\n
            ----\n
            in_MAP: (dtype: np.array): numpy array for the image. 

            '''
            ele_cnt=in_data.shape[0]
            c=1#8
            r=1#ele_cnt/c
            fig=make_subplots(rows=r,cols=c)
            #fig=px.imshow(in_data[0].detach().numpy(),facet_col=0)
            #fig.show()
            # Visualise all the map
            for ele_r in range (r):
                for ele_c in range (c):
                    #im_trans=in_data[(ele_r*c)+ele_c].detach().numpy()[0,:,:]
                    #im_trans=in_data[(ele_r*c)+ele_c][0,:,:]
                    fig.add_trace(go.Heatmap(z=in_data),ele_r+1,ele_c+1)
            # visualise the map progressively.
            fig.show()

        def _AveMAP(in_MAP):
            '''
            Call function to calculate the average attention map

            Arg:\n
            ----\n
            in_MAP: (dtype: np.array): numpy array for the image. 

            return:\n
            -------\n
            buff: (dtype: np.array): numpy array for the average image. 
            '''
            ele_cnt=in_MAP.shape[0]
            buff=np.zeros((in_MAP.shape[2],in_MAP.shape[3]))
            for e in range (ele_cnt):
                buff=buff+in_MAP[e][0,:,:]
            
            self.AvgMAP_perBatch.append(buff*1/ele_cnt)
            return buff
        self.AvgMAP_perBatch=[]
        self.AccAvgMAP=[]
        MapDEC2=nn.Linear(in_features=512,out_features=self.inData_SZ)
        attMAP=Func.relu(MapDEC2(in_MAP.float()))
        a=attMAP.reshape((-1, 1, self.Get_IMG_SZ[1], self.Get_IMG_SZ[2]))
        b=_AveMAP(a.detach().numpy())
        total_map=np.zeros((b.shape[0],b.shape[1]))    
        if len(self.AvgMAP_perBatch)>1:
            for int,X in enumerate(self.AvgMAP_perBatch):
                total_map=total_map+X
            self.AccAvgMAP.append(total_map*(1/(int+1)))
        else:
            self.AccAvgMAP.append(self.AvgMAP_perBatch)
        _plotIMG(b)
        #pass

    def _plotAttMAP(self):
            '''
            Call function to plot the attention map

            Arg:\n
            ----\n
            in_MAP: (dtype: np.array): numpy array for the image. 

            '''
            ele_cnt=len(self.AccAvgMAP)
            c=5#8
            r=ele_cnt/c
            fig=make_subplots(rows=int(r),cols=int(c))
            #fig=px.imshow(in_data[0].detach().numpy(),facet_col=0)
            #fig.show()
            # Visualise all the map
            for ele_r in range (r):
                for ele_c in range (c):
                    #im_trans=in_data[(ele_r*c)+ele_c].detach().numpy()[0,:,:]
                    #im_trans=in_data[(ele_r*c)+ele_c][0,:,:]
                    fig.add_trace(go.Heatmap(z=self.AccAvgMAP[(ele_r*c)+ele_c]),ele_r+1,ele_c+1)
            # visualise the map progressively.
            fig.show()


    def _channeling(self,x_data):
        '''
        Call function use to define the activation function of the layers and channel the input data to the network.\n
    
        Argument:\n
        ---------\n
        param IN: x_data :(dtype: np.array - tensor): stack array of the data. 
        '''

        # Fit the data to the encoder.
        #dataX=Func.relu(self.nn_in(x_data))
        #self.nn_END1(dataX).view
        # flatten the input by changing the shape
        #x_data_f=x_data.reshape(-1,x_data.shape[1]*x_data.shape[2]*x_data.shape[3])
        x_data_f=tr.flatten(x_data,start_dim=1)
        to_latent=self.encoder(x_data_f) # check the latent space
        get_Latent=self.Latent(to_latent)
        Get_latentmuLogVar=self.LatentmulogVar(to_latent)
        #get_LatentLogVar=self.LatentlogVar(to_latent)
        #Cal_kl_loss=(-0.5*(1+get_LatentLogVar - get_Latentmu**2 - 
        #                tr.exp(get_LatentLogVar)).sum(dim=1)).mean(dim=0)
        #reconn_loss=
        get_Repara=self._ParamLatent(Get_latentmuLogVar)

        # Fit the reparameters to decoder for reconstruction.
        get_decoder_data=self.decoder(get_Repara['repara'])
        return {'kl':get_Repara['kl'],'mu':Get_latentmuLogVar,'NN_out':get_decoder_data,'LatentSpace':get_Latent}    
        #pass

    def _ParamLatent(self,muLogVar):
        '''
        Call function use to determine the sampling mean parameter from the latent space to allow 
        gradients to backpropagate from the stochastic part of the model \n 

        Argument:\n
        ---------\n
        param IN: mu (dtype: float): Encoder's latent space mean variables.\n
        param IN: LogVar (dtype: float): Encoder's latent space log variance variables.\n

        param RETURN: KL_loss (dtype: float) KL-divergence loss.\n
        param RETURN: repara (dtype: float) parameter to be send to decoder.\n
        '''
        # Calculate the standard dev
        sigma = tr.exp(0.5*muLogVar) #sigma
        eps = tr.distributions.Normal(0, 1).sample(muLogVar.shape)#tr.randn(size=(muLogVar.size(0),muLogVar.size(1)))
        repara=muLogVar+(eps*sigma)
        self.KL= (sigma**2 + muLogVar**2 - tr.log(sigma) - 1/2).sum()
        return {'kl':self.KL,'eps':eps,'repara':repara}

    def _modeltrainArg(self):
        '''
        Call funtion use to create the parser for training parameters.\n

        Argument:\n
        ---------\n
        param RETURN: vars: (dtype: parser): store the training parameters.\n
        '''
        parser=ap.ArgumentParser()
        parser.add_argument('-e','--epochs',default=30,type=int,help='Number of training epochs.')
        parser.add_argument('-l','--learning rate',default=0.01,type=float,help='the learning rate for the network.')
        
        return vars(parser.parse_args())

    def _trainStep(self,mdl,arg,in_data):
        '''
        Call function to train the network (every) iteration.\n

        Arguments:\n
        ----------\n
        param RETURN: NN_loss: (dtype: float): total loss per iteration .\n

        '''
        optim=opt.Adam(mdl.parameters(),lr=arg['learning rate'])
        lr_schder=opt.lr_scheduler.ExponentialLR(optim,gamma=0.1)
        for X_ep in range (0,arg['epochs']):
            for x,y in in_data:
                optim.zero_grad()
                NNparam=self._channeling(x)#in_data
                    #kl_loss = (-0.5*(1+NNparam['logVar'] - NNparam['mu']**2 - tr.exp(NNparam['logVar'])).sum(dim=1)).mean(dim=0)
                # calculate construction loss.
                Constr_loss=(((x-NNparam['NN_out'].reshape(-1,x.shape[1],x.shape[2],x.shape[3]))**2).sum())#*1/len(x)
                #kl_loss=0
                #recon_loss_criterion = nn.MSELoss()
                #recon_loss = recon_loss_criterion(x, NNparam['NN_out'].reshape(-1,x.shape[1],x.shape[2],x.shape[3]))
                NN_loss=Constr_loss+self.KL
                NN_loss.backward()
                optim.step()
                logging.info('loss per iteration:{}'.format(str(NN_loss)))
            lr_schder.step()
        return {'Model':mdl,'LatenSpaceVar':NNparam['LatentSpace']}

class Dataloader(Dataset):
    """
    Constructs a Dataset to be parsed into a DataLoader
    """
    def __init__(self,X,y):
        X = tr.from_numpy(X).float()

        #Transpose to fit dimensions of my network
        #X = tr.transpose(X,0,1)

        y = tr.from_numpy(y).float()
        self.X,self.y = X,y

    def __getitem__(self, i):
        return self.X[i],self.y[i]

    def __len__(self):
        return self.X.shape[0]    
    
    def _batch(in_data,batch_sz=100):
        '''
        Call function to transform the input data into batch package.

        Argument:\n
        ---------\n
        param IN: in_data (dtype:np.array): stack array of data (1d,2d or 3d data).\n
    
        param IN: batch_sz (dtype: integer): the size of the batch.\n

        Return:\n
        -------\n
        param RETURN: dataloader (dtype: object): Pytorch utils dataloader.\n

        '''

        dataloader=DataLoader(in_data,batch_size=batch_sz,shuffle=True)

        return dataloader

class BootstrapMdl():
    def __init__(self,model,in_data):
        self.mdl=model
        self.data=in_data
    
    def _plotscat(self):    
        # get the latent measurement for every data batch
        fig=go.Figure()
        for ind, (x_fit,y_fit) in enumerate (self.data):
            repara,_=self.mdl._forwardingENDCODER(x_fit) # fit the data through the encoder-decoder; The latent is provided in the result variables
            b=y_fit.detach().numpy().transpose()
            latentVAR=repara.detach().numpy() #convert the tensor to numpy array
            fig.add_traces(go.Scatter(x=latentVAR[:,0],
                                      y=latentVAR[:,1],
                                      mode='markers',
                                      marker_color=b,
                                      text=b,
                                      marker=dict(
                                          size=16,
                                          showscale=True
                                      )))
            #plt.scatter(latentVAR[:, 0], latentVAR[:, 1], c=b, cmap='tab10')                
            #fig.show()
            if ind > 100:
                fig.update_layout(showlegend=False)
                fig.show()
                #plt.colorbar()
                break
            pass
        pass

class AttendMAP():
    def __init__(self,feat,repara):
        super(AttendMAP,self).__init__()
        self.feat=feat
        self.repara=repara
    def _getScalar_a(self):
        sum_W_grad,sum_H_grad=[],[]
        w_ele,h_ele=0,0
        feat_scale_a=[]
        v=[]
        Single_row_tensor=np.array([])
        Vert_tensor=np.array([]).reshape(-1,self.repara.shape[1])
        Get_bxy=self.feat.shape # x = width, y=height and b=batch or feature channel.
        for b in range(0,Get_bxy[0]): 
            # Get the repara, Z per batch
            z=self.repara[b].detach().numpy()
            # Get the feat per batch 
            A=self.feat[b].detach().numpy().reshape(1,-1)
            # determine element-wise gradient for each channel.
            for z_c in z:
                for x in range(A.shape[0]):
                    for y in range(A.shape[1]):
                        if A[x][y] != 0:
                            sum_W_grad.append(z_c/A[x][y]) #element-wise gradient.
                    #sum_W_grad=np.array(sum_W_grad).sum()
                    w_ele=np.array(sum_W_grad).sum()
                    sum_H_grad.append(w_ele)   
                h_ele=np.array(sum_H_grad).sum()
                _a=(1/((x+1)*(y+1)))*h_ele
                Single_row_tensor=np.hstack([Single_row_tensor,_a])
                sum_W_grad,sum_H_grad=[],[]
            Vert_tensor=np.vstack([Vert_tensor,Single_row_tensor.reshape(1,-1)])
            #Vert_tensor=np.concatenate([Vert_tensor,Single_row_tensor],axis=1)
            Single_row_tensor=np.array([]) # clear the array
            #for x in Get_bxy[1]:
            #    for y in Get_bxy[2]:
            #        A = self.feat[b][x]
            #        for C in z:
        return {'a_k':Vert_tensor}         

    def _getscaleMap(self,a_k):
        # stop here
        get_a_sz=a_k.shape
        Lat1_S,Lat2_S=[],[]
        sum_tensor1,sum_tensor2=0,0
        for c in range(get_a_sz[1]): # where c is the number of tensor channel
            a_k_1=a_k[:,c]
            a_k_1_t=tr.tensor(a_k_1.reshape(self.feat.shape[0],1))
            a_k_2_t=a_k_1.reshape(1,self.feat.shape[0])
            b=self.feat.detach().numpy()
            Lat1_S.append(a_k_1_t*self.feat)
            Lat2_S.append(tr.tensor(np.dot(a_k_2_t,b)))
        for e in Lat1_S:
            sum_tensor1=sum_tensor1+e
        for e in Lat2_S:
            sum_tensor2=sum_tensor2+e
        return sum_tensor1 ,sum_tensor2
    
    

class VariationalEncoder(nn.Module):
    def __init__(self, in_data,latent_dims):
        super(VariationalEncoder, self).__init__()
        Get_IMG_SZ=in_data.dataset.data.shape
        self.inData_SZ=Get_IMG_SZ[1]*Get_IMG_SZ[2]
        self.linear1 = nn.Linear(self.inData_SZ, 512)
        self.linear2 = nn.Linear(512, latent_dims) # determine the mean
        self.linear3 = nn.Linear(512, latent_dims) # deterimne the sigma
        
        self.N = tr.distributions.Normal(0, 1)
        #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        #self.N.scale = self.N.scale.cuda()
        self.kl = 0
    
    def forward(self, x):
        x = tr.flatten(x, start_dim=1)
        x = Func.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = tr.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - tr.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    def __init__(self,in_data,latent_dims):
        super(Decoder, self).__init__()
        Get_IMG_SZ=in_data.dataset.data.shape
        self.inData_SZ=Get_IMG_SZ[1]*Get_IMG_SZ[2]
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, self.inData_SZ)
        
    def forward(self, z):
        z = Func.relu(self.linear1(z))
        z = tr.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, in_data,latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(in_data,latent_dims)
        self.decoder = Decoder(in_data,latent_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x)
        z = z.detach().numpy() 
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

def train(autoencoder, data, epochs=20):
    optm = opt.Adam(autoencoder.parameters(),lr=0.01)
    lr_schder=opt.lr_scheduler.ExponentialLR(optm,gamma=0.9)
    cnt=0
    for epoch in range(epochs):
        print(epoch)
        for x, y in data:
            cnt+=1
            #print(cnt)
            #x = x.to(device) # GPU
            optm.zero_grad()
            x_hat = autoencoder(x)
            #a=autoencoder.encoder.kl
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            print(loss)
            loss.backward()
            optm.step()
        #print(cnt)
        lr_schder.step()
        cnt=0
    return autoencoder

def _plotData(in_data):
    '''
    ele_cnt=in_data.shape[0]
    c=1#8
    r=1#ele_cnt/c
    fig=make_subplots(rows=r,cols=c)
    #fig=px.imshow(in_data[0].detach().numpy(),facet_col=0)
    #fig.show()
    # Visualise all the map
    for ele_r in range (r):
        for ele_c in range (c):
            #im_trans=in_data[(ele_r*c)+ele_c].detach().numpy()[0,:,:]
            #im_trans=in_data[(ele_r*c)+ele_c][0,:,:]
            fig.add_trace(go.Image(z=in_data),ele_r+1,ele_c+1)
            # visualise the map progressively.
    fig.show()
    '''
    fig=px.imshow(in_data[0,:,:])
    fig.show()

if __name__=='__main__':
    #x=np.array([[1,1,1],[2,2,2]])
    #y=np.array([[1],[2]])
    
    data = tr.utils.data.DataLoader(
        trv.datasets.MNIST('./data', 
               transform=trv.transforms.ToTensor(), 
               download=True),
        batch_size=128,
        shuffle=True)
    
    # Get data from web or any source.
    
    '''
    a=_Data()
    GetIMG=a._GetData()
    # Process the data
    xTrg_data=a._transform(GetIMG['X_trg'])
    #YTrg_data=a._transform(GetIMG['Y_trg'])
    YTrg_data=GetIMG['Y_trg']
    # pack to Pytorch data loader 
    d_trg=Dataloader(xTrg_data,YTrg_data)
    #d_trg=Data(GetIMG['X_trg'],GetIMG['Y_trg'])
    d_batch_trg=Dataloader._batch(in_data=d_trg)
    '''
    # set up the VAE and train the model
    '''
    VAE = VariationalAutoencoder(data,2)

    # mask off to test the plotting of latent space.
    #Ten_data=a._transform(GetIMG['X_trg'])
    mdl=_CLinearVAE(data,2)
    mdl._NNconfig()
    trg_param=mdl._modeltrainArg()
    c=mdl._TrainStep2(mdl,trg_param,data)
    mdl._plotAttMAP()
    
    #C1 = train(VAE, data)
    
    #C1=mdl._trainStep(VAE,trg_param,data)
    #model_store='/Users/ahum/Documents/[Project] PIPE_DEV/model/VAE_pytorch_Cnew2.sav'
    #joblib.dump(c['Model'],model_store)
    #model_store='/Users/ahum/Documents/[Project] PIPE_DEV/model/VAE_pytorch_C1.sav'
    #joblib.dump(C1,model_store)
    '''
    # load the model

    load_PH='/Users/ahum/Documents/[Project] PIPE_DEV/model/VAE_pytorch_Cnew2.sav'
    mld=joblib.load(load_PH)
    #a=_CLinearVAE(data,2)
    #Bootstrap the model.
    # sort the data to get '1',
    for x,y in data:
        print('label:{}'.format(str(y[0])))
        _plotData(x[0])
        repara,kl,D_flow1=mld._forwardingENDCODER(x[0])
        attentMAP=mld._attentionMAP(D_flow1,repara)
        OP=mld._forwardingDECODER(repara)
        _plotData(OP.detach().numpy()[0])
        #end_X=mld.encoder(x)
        pass
    #BS=BootstrapMdl(mld['Model'],data)
    #BS._plotscat()
    #plot_latent(mld,data)

    


    pass