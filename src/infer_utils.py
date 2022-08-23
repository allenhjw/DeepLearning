# _infer
# function use to infer predict using pre-trained model
# 

# Dependencies
import torch.nn as nn
import torch as tr
from torch.utils.data import DataLoader


class _infer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _ModelInfer(self,trg_mdl,infereData):
        '''
            Function uses to infer the estimation class using the in_data

            Input arg:\n
            ----------\n
            trg_mdl (dtype: model): Pre-trained model.\n
            in_data (dtype: data tensor): input data to infer.\n

        '''
        infereData=DataLoader(infereData,batch_size=64,shuffle=True)
        for x,y in infereData:
            predictOutEng=trg_mdl._forward(x)
            _, predicted = tr.max(predictOutEng, 1)
            
            pass
