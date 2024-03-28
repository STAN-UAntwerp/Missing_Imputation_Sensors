import pandas as pd
from pathlib import Path
import numpy as np

def save_imputed_data(imputed_data, method_name, dataset, missings_percentage):
    
    """
    Save the imputed (and transformed!) data to a csv file.
    """
    
    # set the file location and make sure the folders actually exist
    results_path = Path(__file__).parent.parent.resolve() / 'results' / dataset / method_name
    Path(results_path).mkdir(parents=True, exist_ok=True)

    imputed_data.to_csv(f"{results_path}/{missings_percentage}_imputed.csv", index=False)