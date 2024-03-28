from scipy.interpolate import CubicSpline
import numpy as np

def apply_cubic_spline(df_missings,df_ground_truth):
    """
    Fills in missing values according to a cubic spline of the time series data for each 'id' or longitude, latitude group.

    Parameters:
    - dataframe: input dataframe with the columns (time, target, longitude, latitude)

    Returns:
    - pandas DataFrame with the imputed values
    """
    def cubic_spline_group(group):
        if np.isnan(group['target'].iloc[0]) or np.isnan(group['target'].iloc[-1]):
            avg_target = group['target'].mean()
            group['target'].iloc[0] = avg_target
        if np.isnan(group['target'].iloc[-1]):
            avg_target = group['target'].mean()
            group['target'].iloc[-1] = avg_target
        group_start = group.dropna(subset=['time', 'target'])
        spline_output = CubicSpline(group_start['time'], group_start['target'], bc_type='not-a-knot')
        group['target'] = spline_output(group['time'])
        return group
    
    df_missings = df_missings.sort_values(by=['longitude','latitude','time'], ascending=[False,False,True]).reset_index(drop=True)
    df_output = df_missings.groupby(['longitude', 'latitude'], group_keys=False).apply(cubic_spline_group)
    return df_output

def apply_spline(df_missings,df_ground_truth):
    """
    Fills in missing values according to a spline of the time series data for each 'id' or longitude, latitude group.

    Parameters:
    - dataframe: input dataframe with the columns (time, target, longitude, latitude)

    Returns:
    - pandas DataFrame with the imputed values
    """
    def spline_group(group):
        group_start = group.dropna(subset=['time', 'target'])
        group['target'] =  np.interp(group['time'], group_start['time'], group_start['target'])
        return group
    
    df_missings = df_missings.sort_values(by=['longitude','latitude','time'], ascending=[False,False,True]).reset_index(drop=True)
    df_output = df_missings.groupby(['longitude', 'latitude'], group_keys=False).apply(spline_group)
    return df_output