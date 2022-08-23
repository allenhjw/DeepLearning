# src ultis to create CNN layer based on user define parameters and input data size.

# import API/libraries. 
import numpy as np
import pandas as pd 
import tensorflow as tf
import logging
import sys


# system setup
sys.path.append('/Users/ahum/Documents/[Project] PIPE_DEV/src')
sys.path.append('/Users/ahum/Documents/[Project] PIPE_DEV/data/')
logging.basicConfig(filename='/Users/ahum/Documents/[Project] PIPE_DEV/src/scr_ultis.log', 
                    encoding='utf-8',filemode='a',
                    format='<%(asctime)s> ---- %(message)s', level=logging.INFO)
# print version
logging.info('---- API/Package Versions -----')
logging.info('| numpy | Version:{}'.format(np.__version__))
logging.info('| pandas | Version:{}'.format(pd.__version__))
logging.info('| tensorflow | Version:{}'.format(tf.__version__))
logging.info('| logging | Version:{}'.format(logging.__version__))
logging.info('-------------------------------')

# side note: start define the class
# 1) Data preparing class - Need to create another ultis file and import into here.
# 2) DL define class

class _CNN_Ultis():
    def __init__(self,in_ary='',pad_mode=False,pad_sz=[1,1],filter_sz=[3,3],stride=1):
        '''
        Initialisation for creating CNN baseline NN.\n

        Input:\n
        ------\n
        in_ary: (dtype: array):  input array that store the input data. Default is empty\n
        pad_mode: (dtype: boolean): input array that store the input data. Default is false, which indicate no padding.\n
        pad_sz: (dtype: array): input array indicates the padding size to be added to the input data. Default is [1,1] meaning an addition 1 row and 1 column of pixel is added to the data. pad_sz is ignored if pad_mode is False\n
        filter_sz (dtype: array): input array is the filter size. Default size is [3,3], i.e., 3x3. 
        stride (dtype: integer): the integer indicates the sliding step of the filter across the input data. Default is 1. Do not that the stride must not be more that the filter size\n 
        '''
        try:
            self.in_data=in_ary
            self.pad_mode=pad_mode
            self.padSZ=pad_sz
            self.filterSZ=filter_sz
            self.stride=stride
            logging.info('Ultis initialised.')
            logging.info('------------------')
            if pad_mode==False:
                logging.info('|Padding: {}|filter_sz: {}|stride: {}|'.format(str(pad_mode),str(filter_sz),str(stride)))
            else:
                logging.info('|Padding_sze: {}|filter_sz: {}|stride: {}|'.format(str(pad_sz),str(filter_sz),str(stride)))               
            
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)
        
    def _Cal_NN_SZ(self):
        # get the size of the image
        sh=self.in_data.shape
        self.im_W=sh[1]
        self.im_ht=sh[2]
        self.im_ch=sh[3]

    def _define_NN(self):
        #in_sh=(4,28,28,3)
        #x=tf.random.normal(in_sh)
        #in_shape=[512,512]
        logging.info('-------Defining the CNN structure-----.')
        opti_filterSZ=self.in_data.shape[1]*self.in_data.shape[2]*self.in_data.shape[3]
        #init=tf.keras.initializers.GlorotNormal(seed=None) # initialise the weight with uniform distibution
        
        layer_in=tf.keras.layers.Input(shape=[self.in_data.shape[1],self.in_data.shape[1],self.in_data.shape[-1]],name='CNN_in')
        layer_l1=tf.keras.layers.Conv2D(filters=opti_filterSZ,kernel_size=(self.filterSZ[0],self.filterSZ[1]),padding='same',
                    activation='relu',strides=self.stride,name='CNN_L1',input_shape=self.in_data.shape[1:])(layer_in)
        layer_l1_maxpool=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(layer_l1) #maxpooling with pool_size of 2,2 will reduce the conv size by half. 
        
        layer_l2=tf.keras.layers.Conv2D(filters=opti_filterSZ,kernel_size=(self.filterSZ[0],self.filterSZ[1]),padding='same',
                    activation='relu',strides=self.stride,name='CNN_L2',input_shape=self.in_data.shape[1:])(layer_l1_maxpool)
        layer_l2_maxpool=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(layer_l2) #maxpooling with pool_size of 2,2 will reduce the conv size by half.

        featureMAP=tf.keras.layers.Conv2D(filters=opti_filterSZ,kernel_size=(self.filterSZ[0],self.filterSZ[1]),padding='same',
                    activation='relu',strides=self.stride,name='featureMAP',input_shape=self.in_data.shape[1:])(layer_l2_maxpool)
        layer_Fm_maxpool=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(featureMAP) #maxpooling with pool_size of 2,2 will reduce the conv size by half.    

        layer_Flat=tf.keras.layers.Flatten()(layer_Fm_maxpool)
        layer_D1=tf.keras.layers.Dense(2,activation='sigmoid')(layer_Flat)
        output=tf.keras.layers.Dropout(0.5)(layer_D1)
        # Can include a dropout layer here

        # fit NN to the model
        self.CNN_model=tf.keras.Model(layer_in,output,name='CNN_out')
        self.CNN_model.summary(print_fn=logging.info)
        #logging.info(self.CNN_model.summary())
        pass

if __name__=='__main__':
    # for testing purpose
    from tensorflow.keras import datasets
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    a=_CNN_Ultis(in_ary=train_images)
    a._Cal_NN_SZ()
    a._define_NN()
    pass


