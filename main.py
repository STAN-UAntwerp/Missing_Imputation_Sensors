import logging
import numpy as np
import pandas as pd
from methods.meanImputation import mean_imputation
from methods.meanImputation_nopivot import mean_imputation_no_pivot
from methods.splineImputation import apply_cubic_spline, apply_spline
from methods.desmImputation import DESM_imputer
from methods.soft_matrix_Imputation import soft_impute
from methods.knnImputer import KnnImputer, knn_imputer_function
from methods.mfImputation import mf_imputation
from methods.miceImputation import mice_imputation
from methods.AKE_imputation import AKE_imputation
from utils.load_data import transform_data, load_missing_data, load_imputed_data, load_ground_truth_data
from utils.save_results import save_imputed_data
from utils.evaluation import calculate_metrics_random_missings, calculate_metrics_real_missings, calculate_all_metrics, simple_eval

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # dataset, target, x_loc, y_loc = 'IN', 'temperature', 'x_location', 'y_location'
    dataset, target, x_loc, y_loc = 'CN', 'temp_top', 'longitude', 'latitude'
    df_ground_truth = load_ground_truth_data(dataset)
    df_ground_truth = transform_data(df_ground_truth, 'time', target, x_loc, y_loc)

    # run the imputation methods for different missing amounts

    missings_percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # missings_percentages = [0.05]

    for missing_percentage in missings_percentages:

        df_missings = load_missing_data(dataset, missing_percentage)
        df_missings = transform_data(df_missings, 'time', target, x_loc, y_loc)

        # AKE imputation
        #df_imputed = AKE_imputation(df_missings, k=5, dataset=dataset)
        # save_imputed_data(df_imputed, 'AKE_imputation', dataset, missing_percentage)
        #simple_eval('AKE imputation', missing_percentage, df_missings, df_imputed, df_ground_truth)

        # # mean imputation
        # df_imputed = mean_imputation(df_missings)
        # save_imputed_data(df_imputed, 'mean_imputation', dataset, missing_percentage)
        # simple_eval('Mean imputation', missing_percentage, df_missings, df_imputed, df_ground_truth)

        # # mean imputation without pivot
        # df_imputed = mean_imputation_no_pivot(df_missings)
        # save_imputed_data(df_imputed, 'mean_imputation_no_pivot', dataset, missing_percentage)
        # simple_eval('Mean imputation without pivot', missing_percentage, df_missings, df_imputed, df_ground_truth)

        # # spline imputation
        # df_imputed = apply_spline(df_missings)
        # save_imputed_data(df_imputed, 'spline_imputation', dataset, missing_percentage)
        # simple_eval('Spline imputation', missing_percentage, df_missings, df_imputed, df_ground_truth)

        # # cubic spline imputation
        # df_imputed = apply_cubic_spline(df_missings)
        # save_imputed_data(df_imputed, 'cubic_spline_imputation', dataset, missing_percentage)
        # simple_eval('Cubic Spline imputation', missing_percentage, df_missings, df_imputed, df_ground_truth)

        # # soft imputation (scaled)
        # df_imputed = soft_impute(df_missings, scaled=True)
        # save_imputed_data(df_imputed, 'soft_imputation_scaled', dataset, missing_percentage)
        # simple_eval('Soft imputation (scaled)', missing_percentage, df_missings, df_imputed, df_ground_truth)

        # # soft imputation (not scaled)
        # df_imputed = soft_impute(df_missings, scaled=False)
        # save_imputed_data(df_imputed, 'soft_imputation_not_scaled', dataset, missing_percentage)
        # simple_eval('Soft imputation (not scaled)', missing_percentage, df_missings, df_imputed, df_ground_truth)

        # # knn imputation
        # df_imputed = knn_imputer_function(df_missings)
        # save_imputed_data(df_imputed, 'knn_imputation', dataset, missing_percentage)
        # simple_eval('KNN imputation', missing_percentage, df_missings, df_imputed, df_ground_truth)

        # # # DEMS imputation
        df_imputed = DESM_imputer(df_missings, distance_metric='haversine')
        print(df_imputed)
        # save_imputed_data(df_imputed, 'DEMS_imputation', dataset, missing_percentage)
        # simple_eval('DEMS imputation', missing_percentage, df_missings, df_imputed, df_ground_truth)

        # MICE imputation (!!! takes very long to run)
        # df_imputed = mice_imputation(df_missings)
        # save_imputed_data(df_imputed, 'MICE_imputation', dataset, missing_percentage)
        # simple_eval('MICE imputation', missing_percentage, df_missings, df_imputed, df_ground_truth)

        # # matrix factorization imputation
        # df_imputed = mf_imputation(df_missings)
        # save_imputed_data(df_imputed, 'mf_imputation', dataset, missing_percentage)
        # simple_eval('Matrix Factorization imputation', missing_percentage, df_missings, df_imputed, df_ground_truth)