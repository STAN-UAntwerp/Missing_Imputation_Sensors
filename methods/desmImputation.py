from utils.distance_calc import calculate_closest_points
import warnings
import numpy as np
import pandas as pd
warnings.simplefilter("ignore", category=FutureWarning)

def replace_nan_target(row):
    if np.isnan(row['target']):
        if not np.isnan(row['fill_time']):
            return (1 - row['neighbor_corr']) * row['fill_time'] + row['neighbor_corr'] * row['fill_neighbor']
        else:
            return row['fill_neighbor']
    else:
        return row['target']
    
def read_in_distances():
    df_distances = pd.read_csv('./data/CN_15/sensor_distances.csv')
    return df_distances

def DESM_imputer(df_missings,df_ground_truth,distance_metric="haversine"):
    """
    Apply the DESM imputer according to the method of "Li, Y., Ai, C., Deshmukh, W. P., & Wu, Y. (2008, June). 
    Data estimation in sensor networks using physical and statistical methodologies." 

    Parameters:
    - dataframe: input dataframe with the columns (time, target, longitude, latitude)
    - distance_metric: the distance metric used for the calculations (either haversine or euclidean)
    
    Returns:
    - dataframe with imputed values for the target
    """
    def DESM_single_sensor(sensor,df_distances,dataframe,df_sensor_ids):
        sensor['fill_time'] = sensor['target']
        sensor['fill_neighbor'] = sensor['target']
        sensor['neighbor_corr'] = 0
        sensor['fill_time'].fillna(method='ffill', inplace=True) 
        df_ids = df_sensor_ids.loc[(df_sensor_ids['latitude'] == sensor.iloc[0]['latitude']) &
                    (df_sensor_ids['longitude'] == sensor.iloc[0]['longitude']), :]
        result = df_distances[str(df_ids.index[0])].nsmallest(11)
        rows_with_na = sensor[sensor.isna().any(axis=1)]
        for i, row in rows_with_na.iterrows():
            for i,value in result.items():
                if value == 0:
                    continue
                closest_data = dataframe.loc[(dataframe['latitude'] == df_sensor_ids.loc[i, 'latitude']) &
                    (dataframe['longitude'] == df_sensor_ids.loc[i, 'longitude']), :]
                if not pd.isna(closest_data.loc[closest_data['time'] == row['time'], 'target'].values[0]):
                    sensor.loc[sensor['time']==row['time'],'fill_neighbor'] = closest_data.loc[closest_data['time'] == row['time'], 'target'].values[0]
                    sensor['neighbour'] = closest_data['target'].values
                    sensor.loc[sensor['time']==row['time'],'neighbor_corr'] = sensor['target'].corr(sensor['neighbour'])                    
                    break
        sensor['new_target'] = sensor.apply(replace_nan_target, axis=1)
        return sensor

    unique_comb = df_missings[['latitude', 'longitude']].drop_duplicates()
    df_distances = pd.read_csv('./data/CN_daily_Final/sensor_distances_hv.csv', index_col=0)
    df_sensor_ids = pd.read_csv('./data/CN_daily_Final/sensor_ids.csv', index_col=0)
    df_sensor_ids = df_sensor_ids[df_sensor_ids['longitude'].isin(unique_comb['longitude'])]
    df_sensor_ids = df_sensor_ids[df_sensor_ids['latitude'].isin(unique_comb['latitude'])]
    indexs = df_sensor_ids.index.tolist()
    index = [str(num) for num in indexs]
    df_distances = df_distances.loc[indexs,index]
    imputed_df = df_missings.groupby(['longitude', 'latitude'], group_keys=False).apply(DESM_single_sensor,df_distances,
                                                                                       df_missings,df_sensor_ids)
    imputed_df['target'] = imputed_df['new_target']
    imputed_df.drop(columns=['neighbour','neighbor_corr','fill_neighbor','fill_time','new_target'],inplace=True)
    return imputed_df