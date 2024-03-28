import pandas as pd
from sklearn.impute import KNNImputer
from methods.base import Imputer

class KnnImputer(Imputer):

    def __init__(self, dataset: str, missing_amount: str):
        super().__init__(dataset, missing_amount)
        self.method = 'knn'

    def fit(self, X, y=None):
        """
        Fit the imputer on the data.
        """
        X = self.date_string_to_float(X)
        self.imputer = KNNImputer(n_neighbors=5, weights='distance', metric='nan_euclidean')
        self.imputer.fit(X)
        return self

    def impute(self, X):
        """
        Impute the data.
        """
        X_imputed = self.imputer.transform(X)
        self.imputed_data = pd.DataFrame(X_imputed, columns=X.columns)
        return X_imputed
    

def knn_imputer_function(df_missings,df_ground_truth, **kwargs):

    """
    Fills in missing values according to a knn imputer

    Parameters:
    - dataframe: input dataframe with the columns (time, target, longitude, latitude)

    Returns:
    - pandas DataFrame with the imputed values
    """
    n_neighbors = kwargs.get('n_neighbors', 5) 

    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance', metric='nan_euclidean')
    df_imputed = imputer.fit_transform(df_missings)
    df_imputed = pd.DataFrame(df_imputed, columns=df_missings.columns)

    return df_imputed

