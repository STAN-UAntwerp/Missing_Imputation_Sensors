from missforest.missforest import MissForest

def missforest_imputation(df_missings,df_ground_truth):

    """
    :param dataframe: pandas dataframe with missing values

    :return: pandas dataframe with imputed values
    """
    mf = MissForest()
    mf.fit(
        x=df_missings
    )
    df_imputed_pivot = mf.transform(x=df_missings)
    return df_imputed_pivot
    