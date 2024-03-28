import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.load_data import load_missing_data, load_imputed_data, load_ground_truth_data

def get_mask_for_metrics(data_with_missing: pd.DataFrame, imputed_data: pd.DataFrame) -> np.ndarray:

    """
    Returns a mask array that can be used to calculate the rmse and mae.
    It filters out the values where the data_with_missing is NaN (those that we want to impute),
    and the values where the imputed_data is NaN (those that we were not able to impute).
    """
    data_with_missing_mask = data_with_missing.isnull()
    data_with_missing_mask_array = data_with_missing_mask.values.flatten()
    imputed_mask = ~imputed_data.isnull()
    imputed_mask_array = imputed_mask.values.flatten()
    mask_array = data_with_missing_mask_array & imputed_mask_array

    return mask_array


def calculate_rmse(data_with_missing: pd.DataFrame, imputed_data: pd.DataFrame, ground_truth_data: pd.DataFrame) -> float:
    """
    Calculates the root mean squared error between the imputed_data and the ground_truth_data.
    Only the values where the data_with_missing is NaN are considered.
    When a value is NaN in the imputed_data, it is not considered in the calculation as it was not succesfully imputed.
    """

    mask_array = get_mask_for_metrics(data_with_missing, imputed_data)
    imputed_array = imputed_data.values.flatten()
    ground_truth_array = ground_truth_data.values.flatten()
    rmse = mean_squared_error(imputed_array[mask_array], ground_truth_array[mask_array], squared=False)

    return rmse


def calculate_mae(data_with_missing: pd.DataFrame, imputed_data: pd.DataFrame, ground_truth_data: pd.DataFrame) -> float:
    """
    Calculates the mean absolute error between the imputed_data and the ground_truth_data.
    Only the values where the data_with_missing is NaN are considered.
    When a value is NaN in the imputed_data, it is not considered in the calculation as it was not succesfully imputed.
    """

    mask_array = get_mask_for_metrics(data_with_missing, imputed_data)

    imputed_array = imputed_data.values.flatten()
    ground_truth_array = ground_truth_data.values.flatten()

    mae = mean_absolute_error(imputed_array[mask_array], ground_truth_array[mask_array])

    return mae


def calculate_pce(data_with_missing: pd.DataFrame, imputed_data: pd.DataFrame, ground_truth_data: pd.DataFrame) -> float:
    """
    Calculates the Percentage of Cases in which a missing value can be Estimated (PCE).
    """

    n_to_fill_in = data_with_missing.isnull().sum().sum()
    n_not_filled_in = imputed_data.isnull().sum().sum()

    pce = 1 - n_not_filled_in / n_to_fill_in

    return pce


def calculate_all_metrics(data_with_missing: pd.DataFrame, imputed_data: pd.DataFrame, ground_truth_data: pd.DataFrame) -> tuple:

    """
    Calculates the RMSE, MAE and PCE based on the three datasets passed to this function.
    returns a tuple with the three metrics (RMSE, MAE, PCE)
    """

    rmse = calculate_rmse(data_with_missing, imputed_data, ground_truth_data)
    mae = calculate_mae(data_with_missing, imputed_data, ground_truth_data)
    pce = calculate_pce(data_with_missing, imputed_data, ground_truth_data)

    return rmse, mae, pce


def load_datasets_for_evaluation(method: str, dataset: str, n_missings: str) -> tuple:

    """
    Loads the data_with_missing, imputed_data and ground_truth_data for a given method and number of missings.
    The n_missings can either be a percentage, or a string 'real', for a realistic missings scenario.
    """

    data_with_missing = load_missing_data(dataset, n_missings)
    imputed_data = load_imputed_data(method, dataset, n_missings)
    ground_truth_data = load_ground_truth_data(dataset)

    data_with_missing = data_with_missing.sort_values(by=['time','longitude','latitude'], ascending=[True,False,False])
    imputed_data = imputed_data.sort_values(by=['time','longitude','latitude'], ascending=[True,False,False])
    ground_truth_data = ground_truth_data.sort_values(by=['time','longitude','latitude'], ascending=[True,False,False])

    return data_with_missing, imputed_data, ground_truth_data


def calculate_metrics_random_missings(method: str, dataset: str = 'CN') -> pd.DataFrame:

    """
    Calculates the RMSE, MAE and PCE for one method in the case of MCAR missings.
    Finally, it saves the results in a csv file and returns the results as a pd dataframe.
    TODO: test!
    """

    missing_percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    rmse_values, mae_values, pce_values = [], [], []
    for missings_percentage in missing_percentages:

        data_with_missing, imputed_data, ground_truth_data = load_datasets_for_evaluation(method, dataset, missings_percentage)
        rmse, mae, pce = calculate_all_metrics(data_with_missing, imputed_data, ground_truth_data)
        rmse_values.append(rmse)
        mae_values.append(mae)
        pce_values.append(pce)

    results = pd.DataFrame({'missing_percentage': missing_percentages, 'rmse': rmse_values, 'mae': mae_values,
                            'pce': pce_values})
    results_path = Path(__file__).parent.parent.resolve() / 'results' / dataset / method
    results.to_csv(f'{results_path}/results_random_missings.csv', index=False)

    return results


def calculate_metrics_real_missings(methods: list = ['AKE', 'DEMS', 'dLSTM', 'TSNN', 'linreg_KNN'], dataset: str = 'CN') -> pd.DataFrame:

    """
    Calculates the RMSE, MAE and PCE for all methods in the case of realistics missings.
    Finally, it saves the results in a csv file and returns the results as a pd dataframe.
    TODO: test!
    """

    rmse_values, mae_values, pce_values = [], [], []
    for method in methods:

        data_with_missing, imputed_data, ground_truth_data = load_datasets_for_evaluation(method, dataset, 'real')
        rmse, mae, pce = calculate_all_metrics(data_with_missing, imputed_data, ground_truth_data)
        rmse_values.append(rmse)
        mae_values.append(mae)
        pce_values.append(pce)

    results = pd.DataFrame({'method': methods, 'rmse': rmse_values, 'mae': mae_values, 'pce': pce_values})
    results_path = Path(__file__).parent.parent.resolve() / 'results' / dataset
    results.to_csv(f'{results_path}/results_real_missings.csv', index=False)

    return results


def plot_metric_percentages(methods: list, metric: str, dataset: str) -> None:
    """
    Plots the metric for different methods with different amounts of missing data (MCAR) on the x axis
    """

    marker = itertools.cycle(('+', '.', 'o', '*', 's', '^', 'D')) 

    plt.figure(figsize=(10, 6))

    for method in methods:
        results_path = Path(__file__).parent.parent.resolve() / 'results' / dataset
        results = pd.read_csv(f'{results_path}/{method}/results_random_missings.csv')
        plt.plot(results['missing_percentage']*100, results[metric], label=f'{method}', marker=next(marker))

    plt.xlabel('missing percentage (%)')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} for {dataset} dataset')
    plt.legend()
    plt.tight_layout()
    plot_path = Path(__file__).parent.parent.resolve() / 'plots' / dataset
    Path(plot_path).mkdir(parents=True, exist_ok=True) 
    plt.savefig(f'{plot_path}/{metric}_percentages.png')
    plt.savefig(f'{plot_path}/{metric}_percentages.pdf')
    plt.show()


def plot_metric_real_world_missings(methods: list, metric: str, dataset: str) -> None:

    """
    Plots the metric for different methods with different amounts of missing data (MCAR) on the x axis
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        results = pd.read_csv(f'results/results_real_missings.csv')
        plt.bar(i, results['missing_percentage'].values[-1], label=f'{method}')

    ax.set_xticks(range(len(methods)), labels=[f'method {i}' for i in methods])
    plt.xlabel('method')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} for {dataset} dataset (real-world missings)')
    plt.tight_layout()
    plt.savefig(f'plots/{metric}_real.png')
    plt.savefig(f'plots/{metric}_real.pdf')

    plt.show()


def simple_eval(method, percentage_missing, df_missing, imputed_data, df_ground_truth):
    # reorder columns in imputed_data if necessary
    imputed_data = imputed_data[['time', 'target', 'longitude', 'latitude']]
    rmse, mae, pce = calculate_all_metrics(df_missing, imputed_data, df_ground_truth)
    logging.info(f"{method} ({int(percentage_missing*100)}% missing data): RMSE: {rmse:.3f}, MAE: {mae:.3f}, PCE: {pce:.3f}")


def simple_eval_masked(fold,missing_perc,method, df_missing, imputed_data, df_ground_truth):
    imputed_data = imputed_data[['time', 'target', 'longitude', 'latitude']]
    df_missing = df_missing.sort_values(by=['time','longitude','latitude'], ascending=[True,False,False])
    imputed_data = imputed_data.sort_values(by=['time','longitude','latitude'], ascending=[True,False,False])
    df_ground_truth = df_ground_truth.sort_values(by=['time','longitude','latitude'], ascending=[True,False,False])
    rmse, mae, pce = calculate_all_metrics(df_missing, imputed_data, df_ground_truth)
    logger = logging.getLogger("CN_Missings")
    logger.info(f"Iteration {fold: <5} | Missing {missing_perc: <10} | {method : <25} | RMSE:{rmse:.3f} | MAE:{mae:.3f} | PCE:{pce:.3f}")
    return rmse, mae, pce

