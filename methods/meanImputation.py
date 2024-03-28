import pandas as pd

def mean_imputation (df_missings,df_ground_truth):
    """
    Apply mean imputation to the sensor time series data, by calculating the mean of the target values for each available sensor at the given time.

    Parameters:
    - dataframe: input dataframe with the columns (time, target, longitude, latitude)
    
    Returns:
    - dataframe with imputed values for the target
    """
    dataframe = df_missings.copy()
    def mean_imputation(series):
        return series.fillna(series.mean())
    pivot_df = dataframe.pivot(index='time', columns=['longitude', 'latitude'], values='target')
    imputed_df = pivot_df.apply(mean_imputation)
    unpivoted_df = imputed_df.reset_index().melt(id_vars='time', var_name=['longitude', 'latitude'], value_name='target')

    return unpivoted_df