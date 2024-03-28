from fancyimpute import IterativeImputer
import pandas as pd

def mice_imputation(df_missings,df_ground_truth,**kwargs):
    """
    Implementation of the MICE algorithm from:
    M.J. Azur, E.A. Stuart, C. Frangakis, P.J. Leaf, Multiple imputation by
    chained equations: what is it and how does it work?, Int. J. Methods Psychiat. Res.

    Parameters:
    - dataframe: input dataframe with the columns (time, target, longitude, latitude)
    
    Returns:
    - dataframe with imputed values for the target
    """
    n_nearest_features = kwargs.get('n_nearest_features', 10)
    iterativeImputer = IterativeImputer(max_iter=50, tol=0.001,n_nearest_features=n_nearest_features)
    pivot_df = df_missings.pivot(index='time', columns=['longitude', 'latitude'], values='target')
    df_output = iterativeImputer.fit_transform(pivot_df.to_numpy())
    unpivoted_df = pd.DataFrame(df_output, index=pivot_df.index, columns=pivot_df.columns)
    imputed_df = unpivoted_df.reset_index().melt(id_vars='time', var_name=['longitude', 'latitude'], value_name='target')
    return imputed_df