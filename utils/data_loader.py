import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def check_all_nan(df):
    columns_with_all_nan = [col for col in df.columns if df[col].isna().all()]
    return columns_with_all_nan

def checkmax_sequence(df):
    grouped = df.groupby(['latitude', 'longitude'])
    max_sequences = []
    for (lat, long), group in grouped:
        current_sequence = 0
        max_sequence = 0
        for value in group['target']:
            if pd.isna(value):
                current_sequence += 1
            else:
                if current_sequence > max_sequence:
                    max_sequence = current_sequence
                    current_sequence = 0
                else:
                    current_sequence = 0
        if current_sequence > max_sequence:
            max_sequence = current_sequence
        max_sequences.append({'latitude': lat, 'longitude': long, 'max_sequence': max_sequence})
    result_df = pd.DataFrame(max_sequences)
    return result_df

def obtain_data(df_missings,seq_len):

    #Define sequence length and obtain dataframe parameters
    pivot_df_ms_1 = df_missings.pivot(index='time', columns=['longitude', 'latitude'], values='target')
    pivot_df_ms = pivot_df_ms_1.droplevel(level='latitude', axis=1).reset_index(drop=True)
    pivot_df_ms.columns = [str(col) for col in pivot_df_ms.columns]
    for i in range(0, len(pivot_df_ms), round(seq_len/2)):
        chunk = pivot_df_ms.iloc[i:i+round(seq_len/2)]
        cols_nan = check_all_nan(chunk)
        for colum in cols_nan:
            pivot_df_ms.loc[i,colum] = pivot_df_ms.iloc[i].mean()

    scaler_ms = MinMaxScaler(feature_range=(0, 1.00001))
    df_ms = scaler_ms.fit_transform(pivot_df_ms)
    no, dim = pivot_df_ms.shape
    no = no - seq_len 

    #Obtain three dimensional numpy arrays for training - both missings and ground truth
    ms_X= list()
    for i in range(no):
        temp_ms = df_ms[i:(i+seq_len)]
        ms_X = ms_X + [temp_ms]
    x = np.asarray(ms_X,dtype=object).astype('float32')
    
    #Get mask missing
    m = ~np.isnan(x)
    m = m.astype(int)
    
    #Get time array with missings
    t = list()    
    for i in range(no):
        temp_m = m[i,:,:]
        temp_t = np.ones([seq_len, dim])
        for j in range(dim):
            for k in range(1, seq_len):
                if temp_m[k, j] == 0:
                    temp_t[k, j] = temp_t[k-1, j] + 1
        t = t + [temp_t]
    t = np.asarray(t,dtype=object).astype('float32')
    
    shape = x.shape
    x = pd.DataFrame(np.vstack(x))
    x = x.to_numpy()
    x = x.reshape(shape[0], shape[1], shape[2])
    x = np.nan_to_num(x, nan=0)
    return x,m,t,scaler_ms