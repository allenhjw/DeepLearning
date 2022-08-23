# src ultis to create CNN layer based on user define parameters and input data size.

# import API/libraries. 
import numpy as np
import pandas as pd 
import tensorflow as tf
import logging
import sys
# system setup
sys.path.append('/Users/ahum/Documents/[Project] PIPE_DEV/src')
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

