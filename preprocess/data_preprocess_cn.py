#Import of necessary packages 
import pandas as pd
import warnings
import numpy as np
import os
from itertools import product
from pandas import date_range
warnings.simplefilter(action='ignore', category=FutureWarning)

def generate_random_missing_values(target_folder,dataset, missing_percentage, output_name, target_column, seed=42):
    """
    Generates missing values randomly in a specified column of a dataset.

    Parameters:
    - dataset: pandas DataFrame, the input dataset
    - missing_percentage: float, the percentage of missing values to generate (0.0 to 1.0)
    - output_name: str, the name of the CSV file to save the modified dataset
    - target_column: str, the name of the column to introduce missing values
    - seed: int or None, seed for random number generation

    Returns:
    - pandas DataFrame, the dataset with randomly generated missing values in the specified column
    """
    np.random.seed(seed)
    modified_dataset = dataset.copy()
    num_missing_values = int(np.floor(missing_percentage * dataset[target_column].size))
    random_rows = np.random.choice(dataset.shape[0], size=num_missing_values, replace=False)
    modified_dataset.loc[random_rows, target_column] = np.nan
    file_path = os.path.join(target_folder, output_name)
    modified_dataset.to_csv(file_path, index=False)
    return modified_dataset
    
def main(value_to_predict = 'temp_top', per_day = True, limit = 1000):
    
    #Read in the extracted data of both extra info and the temp data
    data_path = os.path.join(os.getcwd(), 'data')
    info_sensors = pd.read_excel(os.path.join(data_path,"Info_Sensors_OK.xlsx"))
    temp_sensors = pd.read_csv(os.path.join(data_path,'CN_Data_Full.csv'))
    del_cols = ['humidity','error','temp_bot','temp_mid','temp_top']
    updated_del_cols = [col for col in del_cols if col != value_to_predict]
    temp_sensors = temp_sensors.drop(columns=updated_del_cols)
    temp_sensors['time'] = pd.to_datetime(temp_sensors['time'])
    
    #Filter sensors based on location and delete those with the same location
    info_sensors['location'] = (
        info_sensors['longitude'].astype(str)+','+ info_sensors['latitude'].astype(str))
    info_sensors_filtered = info_sensors.drop_duplicates(subset='location', keep='first').copy()
    info_sensors_filtered.drop('location', axis=1, inplace=True)
    info_sensors_filtered['cnNumber'].unique()
    unique_cn_numbers = info_sensors_filtered['cnNumber'].unique().tolist()
    temp_sensors_filtered = temp_sensors[temp_sensors['id'].isin(unique_cn_numbers)]
    
    #Drop duplicate values for the same moment in time 
    temp_sensors_filtered = temp_sensors_filtered.drop_duplicates(subset=['id', 'time']).reset_index(drop=True)
    
    #Filter data over time to avoid to many NANs 
    temp_sensors_filtered = temp_sensors_filtered[temp_sensors_filtered['time'] >= '2021-04-12 00:00:00']
    
    #Make complete data for all sensors (e.g. add data when certain data points are missing)
    all_dates = date_range(start='2021-04-12 00:00:00', end='2021-09-30 23:45:00', freq='15min')
    all_ids = temp_sensors_filtered['id'].unique()
    all_combinations = list(product(all_dates, all_ids))
    new_df = pd.DataFrame(all_combinations, columns=['time', 'id'])
    merged_df = pd.merge(new_df, temp_sensors_filtered, how='left', on=['time', 'id'])

    #Make the dataframes + add the locations
    ids_with_missing_values = merged_df.loc[merged_df['temp_top'].isna(), 'id'].unique()
    df_ground_truth =  merged_df[~merged_df['id'].isin(ids_with_missing_values)]
    df_missings = merged_df[merged_df['id'].isin(ids_with_missing_values)]
    info_sensors_filtered_join = info_sensors_filtered[['longitude','latitude','cnNumber']]
    if per_day == True:
        df_missings = df_missings.groupby(['id', df_missings['time'].dt.date])['temp_top'].agg(lambda x: np.nan if any(pd.isna(x)) else np.mean(x)).reset_index()
        df_ground_truth = df_ground_truth.groupby(['id',df_ground_truth['time'].dt.date])['temp_top'].mean().reset_index()
    df_ground_truth = pd.merge(df_ground_truth, info_sensors_filtered_join, how='left', left_on=['id'],right_on=['cnNumber'])
    df_ground_truth = df_ground_truth.drop(columns=['cnNumber'])

    #Drop any sensors with only missings for mask later
    missing_all_ids = df_missings.groupby('id')['temp_top'].apply(lambda x: x.isna().all()).loc[lambda x: x].index.to_list()
    df_missings =  df_missings[~df_missings['id'].isin(missing_all_ids)]
    
    #Limit the number of sensors for faster run times and easier evaluation
    lijst_limit = df_missings['id'].unique().tolist()[0:limit]
    df_missings =  df_missings[df_missings['id'].isin(lijst_limit)]
    lijst_limit = df_ground_truth['id'].unique().tolist()[0:limit]
    df_ground_truth =  df_ground_truth[df_ground_truth['id'].isin(lijst_limit)]
    
    #Save the files 
    if per_day == True:
        target_folder = os.path.join(data_path, "CN_daily_Final_RN")
        target_folder_2 = os.path.join(data_path, "CN_daily_Final")
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        if not os.path.exists(target_folder_2):
            os.makedirs(target_folder_2)
    else:
        target_folder = os.path.join(data_path, "CN_15_Final_RN")
        target_folder_2 = os.path.join(data_path, "CN_15_Final")
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        if not os.path.exists(target_folder_2):
            os.makedirs(target_folder_2)
    file_path = os.path.join(target_folder, "ground_truth.csv")
    df_ground_truth.to_csv(file_path,index=False)
    file_path = os.path.join(target_folder, "missings_all.csv")
    df_missings.to_csv(file_path,index=False)
    file_path_2 = os.path.join(target_folder_2, "ground_truth.csv")
    df_ground_truth.to_csv(file_path_2,index=False)
    file_path_2 = os.path.join(target_folder_2, "missings_all.csv")
    df_missings.to_csv(file_path_2,index=False)

    for missing_perc in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
        generate_random_missing_values(target_folder,df_ground_truth, missing_perc,
                                       '{perc}_missings.csv'.format(perc=str(missing_perc)),'temp_top')
                                                         
if __name__ == "__main__":
    print("Per day")
    main(value_to_predict = 'temp_top', per_day = True, limit = 1500)

