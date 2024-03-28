#Import of necessary packages 
import pandas as pd
import warnings
import numpy as np
import os
import random
warnings.simplefilter(action='ignore', category=FutureWarning)

def randomly_select_elements(data_list, num_elements, seed=42):
    random.seed(seed)
    selected_elements = random.sample(data_list, num_elements)
    return selected_elements

def main(daily=True,relative=True):
    
    #Get data and set seed
    np.random.seed(42)
    parent_directory = os.getcwd()
    if daily == True:
        data_path = os.path.join(parent_directory, 'data')
        data_path = os.path.join(data_path, 'CN_daily_Final')
    else:
        data_path = os.path.join(parent_directory, 'data/CN_15_Final')
    
    df_ground_truth = pd.read_csv(os.path.join(data_path, "ground_truth.csv"))
    df_missings = pd.read_csv(os.path.join(data_path, "missings_all.csv"))
    
    #Calculate relative number of missings based on the actual data 
    #Otherwise replace all values with missings
    if relative == True: 
        number_for_mask = np.round(len(df_ground_truth['id'].unique())*
                                   (len(df_missings['id'].unique())/(len(df_missings['id'].unique())+
                                                    len(df_ground_truth['id'].unique()))))
        selected_elements = randomly_select_elements(list(df_missings['id'].unique()), int(number_for_mask))
        dfsub_missings = df_missings[df_missings['id'].isin(selected_elements)].reset_index(drop=True)
    else:
        dfsub_missings = df_missings
        
    #Select the random missings and shuffle the ground truth for randomization
    unique_ids = df_ground_truth['id'].unique()
    shuffled_ids = np.random.permutation(unique_ids)
    
    #Start a loop to assign a random missing to a random ground truth id
    m = 0
    for u in shuffled_ids:
        if m >= len(dfsub_missings['id'].unique()):
            break
        else:
            dfsub_one_missing = dfsub_missings[dfsub_missings['id'] == dfsub_missings['id'].unique()[m]]
            df_missing_rows = dfsub_one_missing[dfsub_one_missing.isnull().any(axis=1)]
            df_ground_truth.loc[(df_ground_truth['id'] == u) & (df_ground_truth['time'].isin(df_missing_rows['time'])),
                        'temp_top'] = df_missing_rows['temp_top'].values
            m += 1
    df_ground_truth.to_csv(os.path.join(data_path, "masked_missings.csv"))
    
if __name__ == "__main__":
    print("Per day")
    main(daily=True,relative=False)
    print("Per quarter")
    main(daily=False,relative=False)