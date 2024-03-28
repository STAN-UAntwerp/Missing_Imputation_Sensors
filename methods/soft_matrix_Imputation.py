from fancyimpute import SoftImpute, BiScaler
import pandas as pd 

def soft_impute(df_missings,df_ground_truth,scaled=True):
    """
    Implementation of the SoftImpute algorithm from:
    "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
    by Mazumder, Hastie, and Tibshirani.

    Parameters:
    - dataframe: input dataframe with the columns (time, target, longitude, latitude)
    - scaled: whether or not to scale the data 
    
    Returns:
    - dataframe with imputed values for the target
    """
    softImpute = SoftImpute(verbose=False)
    if scaled == True:
        biscaler = BiScaler(verbose=False)
        pivot_df = df_missings.pivot(index='time', columns=['longitude', 'latitude'], values='target')
        for index, row in pivot_df.iterrows():
            if row.isnull().all():
                pivot_df.at[index, pivot_df.columns[0]] = 8
        for column in pivot_df.columns:
            if pivot_df[column].isnull().all():
                neighbor_column = pivot_df.columns[pivot_df.columns.get_loc(column) - 1]
                pivot_df.at[pivot_df.index[0], column] = pivot_df.at[pivot_df.index[0], neighbor_column]
        df_incomplete = biscaler.fit_transform(pivot_df.to_numpy())
        df_output_scaled = softImpute.fit_transform(df_incomplete)
        df_output = biscaler.inverse_transform(df_output_scaled)
    else:
        pivot_df = df_missings.pivot(index='time', columns=['longitude', 'latitude'], values='target')
        df_output = softImpute.fit_transform(pivot_df.to_numpy())
    unpivoted_df = pd.DataFrame(df_output, index=pivot_df.index, columns=pivot_df.columns)
    imputed_df = unpivoted_df.reset_index().melt(id_vars='time', var_name=['longitude', 'latitude'], value_name='target')
    return imputed_df