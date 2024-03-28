from utils.load_data import load_missing_data, load_ground_truth_data
import pandas as pd

class Imputer:

    def __init__(self, dataset: str, missing_amount: str):
        #TODO: add an output path to save the imputed data? This should include the imputation method's name.
        self.dataset = dataset
        self.missing_amount = missing_amount
        self.method = None
        self.imputed_data = None

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        """
        Load the data from the csv files.
        TODO: if we want to drop certain columns, we should probably add this here.
        """
        
        df_missings = load_missing_data(self.dataset, self.missing_amount)
        df_ground_truth = load_ground_truth_data(self.dataset)
        return df_missings, df_ground_truth
    
    def make_subset(self, df_missings: pd.DataFrame, df_ground_truth: pd.DataFrame, percentage: float) -> tuple[pd.DataFrame, pd.DataFrame]:

        """
        Make a subset of the data to fit and impute. 
        """

        n_rows = int(df_missings.shape[0] * percentage)
        df_missings_subset = df_missings[:n_rows]
        df_ground_truth_subset = df_ground_truth[:n_rows]
        return df_missings_subset, df_ground_truth_subset
               
    def fit(self, X, y=None):
        raise NotImplementedError

    def impute(self, X):
        raise NotImplementedError
    
    def save_imputed_data(self):
        
        """
        Save the imputed data to a csv file.
        """
        
        self.imputed_data.to_csv(f"results/{self.dataset}/{self.method}_{self.missing_amount}_imputed.csv", index=False)
    

