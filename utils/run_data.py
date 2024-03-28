from pathlib import Path
from utils.load_data import transform_data, load_missing_data, load_ground_truth_data
from utils.run_utils import run_all_methods_off_tf, run_all_methods_on_tf
import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

def data_run(results_df,missing_percentage,daily=True,randomisatie=False,on_tf=True):
    if daily == True:
        dataset = 'CN_daily_Final'
    else:
        dataset = 'CN_15_Final' 
    if randomisatie == True:
        dataset = dataset + '_RN'
    target, x_loc, y_loc, = 'temp_top', 'longitude', 'latitude'
    df_ground_truth_start = load_ground_truth_data(dataset)
    df_ground_truth_all = transform_data(df_ground_truth_start, 'time', target, x_loc, y_loc)
    df_missings_start = load_missing_data(dataset, missing_percentage)
    df_missings_all = transform_data(df_missings_start, 'time', target, x_loc, y_loc)
    results_path = Path(__file__).parent.parent.resolve() / 'results' / dataset 
    Path(results_path).mkdir(parents=True, exist_ok=True)
    unique_comb = df_ground_truth_all[['longitude', 'latitude']].drop_duplicates()
    num_folds = 5

    if on_tf == True:
        for i in range(2):
            np.random.seed(i+5)
            X_train, X_test = train_test_split(unique_comb, test_size=0.75, random_state=i+5)
            df_ground_truth = df_ground_truth_all[df_ground_truth_all['longitude'].isin(X_test['longitude']) & 
                                df_ground_truth_all['latitude'].isin(X_test['latitude'])]
            df_missings = df_missings_all[df_missings_all['longitude'].isin(X_test['longitude']) & 
                                df_missings_all['latitude'].isin(X_test['latitude'])]
            unique_comb_folds = df_ground_truth[['longitude', 'latitude']].drop_duplicates()
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=i+5)
            for fold, (train_index, val_index) in enumerate(kf.split(unique_comb_folds)):
                X_fold_test_key = unique_comb_folds.iloc[val_index]
                gt_test_df = pd.merge(df_ground_truth, X_fold_test_key, 
                                    on=['longitude', 'latitude'])
                ms_test_df = pd.merge(df_missings, X_fold_test_key, 
                                    on=['longitude', 'latitude'])
                print(ms_test_df.shape)
                print(ms_test_df.shape[0]/172)
                fold = '{} x {}'.format(i+1,fold+1)
                results_df = run_all_methods_on_tf(fold,missing_percentage,results_df,dataset,
                                            ms_test_df,gt_test_df,
                                            missing_percentage,dataset)
    else:
        for i in range(2):
            np.random.seed(i+5)
            X_train, X_test = train_test_split(unique_comb, test_size=0.75, random_state=i+5)
            df_ground_truth = df_ground_truth_all[df_ground_truth_all['longitude'].isin(X_test['longitude']) & 
                                df_ground_truth_all['latitude'].isin(X_test['latitude'])]
            df_missings = df_missings_all[df_missings_all['longitude'].isin(X_test['longitude']) & 
                                df_missings_all['latitude'].isin(X_test['latitude'])]
            unique_comb_folds = df_ground_truth[['longitude', 'latitude']].drop_duplicates()
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=i+5)
            for fold, (train_index, val_index) in enumerate(kf.split(unique_comb_folds)):
                X_fold_test_key = unique_comb_folds.iloc[val_index]
                gt_test_df = pd.merge(df_ground_truth, X_fold_test_key, 
                                    on=['longitude', 'latitude'])
                ms_test_df = pd.merge(df_missings, X_fold_test_key, 
                                    on=['longitude', 'latitude'])
                print(ms_test_df.shape)
                print(ms_test_df.shape[0]/172)
                fold = '{} x {}'.format(i+1,fold+1)
                results_df = run_all_methods_off_tf(fold,missing_percentage,results_df,dataset,
                                            ms_test_df,gt_test_df,
                                            missing_percentage,dataset)
    return results_df