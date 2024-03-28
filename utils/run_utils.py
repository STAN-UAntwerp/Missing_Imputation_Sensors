import time 
import traceback
import pandas as pd
import logging
import json 

#Utils import
from utils.evaluation import simple_eval_masked
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#Methods import
from methods.meanImputation import mean_imputation
from methods.splineImputation import apply_spline
from methods.knnImputer import knn_imputer_function
from methods.desmImputation import DESM_imputer
from methods.AKE_imputation import AKE_imputation
from methods.soft_matrix_Imputation import soft_impute
from methods.miceImputation import mice_imputation
from methods.MCMC import MCMC_imputation
from methods.MRNN.main_mrnn import MRNN_imputation
from methods.MIDA import MIDA_imputation
from methods.BRITS.main_brits import BRITS_imputation
from methods.missforest import missforest_imputation

def run_single_method(fold,missing_perc,plot_name,results_df,
                      method,df_missings,name_save_results,dataset_name,
                      missing_percentage,df_ground_truth,
                      imputation_function, *args, **kwargs):
    logger = logging.getLogger("CN_Missings")
    start_time = time.time()
    with open('utils/best_hyperparam_values.json', 'r') as file:
        hyperparam_data = json.load(file)
    hyperparams = hyperparam_data.get(method, {})
    try:
        df_imputed = imputation_function(df_missings,df_ground_truth, 
                                         *args,**hyperparams, **kwargs)
    except Exception as e:
        logger.critical(f"Error: {e}")
        logger.critical(traceback.format_exc())
    end_time = time.time()  
    execution_time = end_time - start_time 
    rmse, mae, pce = simple_eval_masked(fold,missing_perc,method, df_missings, df_imputed, df_ground_truth)
    results_df = pd.concat([results_df, pd.DataFrame({'Fold': fold,'Missings': missing_perc, 'Method': plot_name, 
                                                      'RMSE': rmse, 'MAE': mae, 'PCE': pce,
                                                      'Execution time':execution_time}, index=[0])], ignore_index=True)
    return results_df


def run_all_methods_off_tf(fold,missing_perc,results_df,dataset_name,df_missings,
                    df_ground_truth,missing_percentage,dataset):
    # # Mean imputation
    # method,name_save_results,imputation_function = 'Mean imputation','mean_imputation',mean_imputation
    # plot_name = 'Mean imputation'
    # results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
    #                                dataset_name,missing_percentage,df_ground_truth,
    #                                imputation_function)
    # # Spline imputation
    # method,name_save_results,imputation_function = 'Spline imputation','spline_imputation',apply_spline
    # plot_name = 'Spline - Kreindler, D. M., & Lumsden, C. J. (2016)'
    # results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
    #                                dataset_name,missing_percentage,df_ground_truth,
    #                                imputation_function)
    # # DESM imputation
    # method,name_save_results,imputation_function = 'DESM imputation','desm_imputation',DESM_imputer
    # plot_name = 'DESM - Gruenwald et al. (2010)'
    # results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
    #                                dataset_name,missing_percentage,df_ground_truth,
    #                                imputation_function)
    # KNN imputation
    method,name_save_results,imputation_function = 'KNN imputation','knn_imputation',knn_imputer_function
    plot_name = 'KNN'
    results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
                                    dataset_name,missing_percentage,df_ground_truth,
                                    imputation_function)
    # # MF/Matrix Completion imputation
    # method,name_save_results,imputation_function = 'MC/MF imputation scaled','MC/MF_imputation_scaled',soft_impute
    # plot_name = 'MC - Mazumder, R., Hastie, T., & Tibshirani, R. (2010)'
    # results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
    #                                 dataset_name,missing_percentage,df_ground_truth,
    #                                 imputation_function)
    # # MICE imputation
    # method,name_save_results,imputation_function = 'MICE imputation','MICE_imputation',mice_imputation
    # plot_name = 'MICE - White, I. R., Royston, P., & Wood, A. M. (2011)'
    # results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
    #                                 dataset_name,missing_percentage,df_ground_truth,
    #                                 imputation_function)
    # # AKE imputation
    # method,name_save_results,imputation_function = 'AKE imputation','AKE_imputation',AKE_imputation
    # plot_name = 'AKE - Pan, L., & Li, J. (2010)'
    # results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
    #                                 dataset_name,missing_percentage,df_ground_truth,
    #                                 imputation_function, dataset=dataset)
    # # MissForest imputation
    # method,name_save_results,imputation_function = 'MissForest imputation','missforest_imputation',missforest_imputation
    # plot_name = 'MissForest - D. J. Stekhoven and P. Bühlmann (2011)'
    # results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
    #                                 dataset_name,missing_percentage,df_ground_truth,
    #                                 imputation_function)
    # # MIDA imputation
    #method,name_save_results,imputation_function = 'MIDA imputation','MIDA_imputation',MIDA_imputation
    #plot_name = 'MIDA - Gondara, L., & Wang, K. (2018)'
    #results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
    #                                 dataset_name,missing_percentage,df_ground_truth,
    #                                 imputation_function, df_gt=df_ground_truth)   
    # # MRNN imputation
    # method,name_save_results,imputation_function = 'MRNN imputation','MRNN_imputation',MRNN_imputation
    # plot_name = 'MRNN - Yoon, J., Zame, W. R., & van der Schaar, M. (2018)'
    # results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
    #                                 dataset_name,missing_percentage,df_ground_truth,
    #                                 imputation_function,df_gt=df_ground_truth)  
    return results_df


def run_all_methods_on_tf(fold,missing_perc,results_df,dataset_name,df_missings,
                    df_ground_truth,missing_percentage,dataset):
    # MCMC imputation
    method,name_save_results,imputation_function = 'MCMC imputation','MCMC_imputation',MCMC_imputation
    plot_name = 'MCMC - Schunk, D. (2008)'
    results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
                                    dataset_name,missing_percentage,df_ground_truth,
                                    imputation_function) 
    # BRITS imputation 
    method,name_save_results,imputation_function = 'BRITS imputation','BRITS',BRITS_imputation
    plot_name = 'BRITS - Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018)'
    results_df = run_single_method(fold,missing_perc,plot_name,results_df,method,df_missings,name_save_results,
                                    dataset_name,missing_percentage,df_ground_truth,
                                    imputation_function,df_gt=df_ground_truth)  
    return results_df