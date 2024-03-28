# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from methods.splineImputation import apply_spline
warnings.filterwarnings("ignore")
import shutil
import os
from utils.data_loader import obtain_data
from methods.MRNN.mrnn import mrnn
import pandas as pd
import numpy as np

def make_final_prediction(df):
  final_imputation = np.empty((df.shape[0]+df.shape[1],df.shape[2]))
  for sensor in range(df.shape[2]):
      for i in range(df.shape[0]):
          if i < df.shape[1]:
              m = i
          else:
              m = df.shape[1] - 1
          data_list = []
          for seq in range(m, -1, -1):
              data_list.append(df[i-seq,seq,sensor])
          data_array = np.array(data_list)
          final_imputation[i, sensor] = np.mean(data_array)       
      final_imputation[i+1:, sensor] = df[i,:,sensor] 
  return final_imputation

def MRNN_imputation(df_missings,df_ground_truth,**kwargs):
  """
    MRNN main function.
  
  Args:
    - seq_len: sequence length of time-series data
    - h_dim: hidden state dimensions
    - batch_size: the number of samples in mini batch
    - iteration: the number of iteration
    - learning_rate: learning rate of model training
    
  Returns:
    - output:
      - imputed_x: imputed data
  """  
  iteration = kwargs.get('iteration', 100)
  batch_size = kwargs.get('batch_size', 1)
  h_dim = kwargs.get('h_dim', 10)
  seq_len = kwargs.get('seq_len', 10)
  learning_rate = kwargs.get('learning_rate', 0.01)
  x,m,t,scaler_ms =  obtain_data(df_missings,seq_len)
  if os.path.exists('tmp/mrnn_imputation'):
    shutil.rmtree('tmp/mrnn_imputation')
  model_parameters = {'h_dim': h_dim,
                      'batch_size': batch_size,
                      'iteration': iteration, 
                      'learning_rate': learning_rate}  
  mrnn_model = mrnn(x, model_parameters)
  mrnn_model.fit(x, m, t)
  imputed_x = mrnn_model.transform(x, m, t)
  imputed_array = np.asarray(imputed_x,dtype=object).astype('float32')
  imputed = make_final_prediction(imputed_array)
  imputed = scaler_ms.inverse_transform(imputed)
  imputed_df = pd.DataFrame(imputed)
  pivot_df_ms = df_missings.pivot(index='time', columns=['longitude', 'latitude'], values='target')
  imputed_df.columns = pivot_df_ms.columns
  imputed_df.index = pivot_df_ms.index
  unpivoted_df = imputed_df.reset_index().melt(id_vars='time', var_name=['longitude', 'latitude'], value_name='target')
 
  return unpivoted_df