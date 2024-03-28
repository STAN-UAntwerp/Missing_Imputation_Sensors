import json
from itertools import product
import logging
import random
from utils.load_data import transform_data, load_missing_data, load_ground_truth_data
from utils.evaluation import simple_eval_masked
from sklearn.model_selection import train_test_split
from methods.knnImputer import knn_imputer_function
from methods.AKE_imputation import AKE_imputation
from methods.miceImputation import mice_imputation
from methods.MRNN.main_mrnn import MRNN_imputation
from methods.MIDA import MIDA_imputation
from methods.BRITS.main_brits import BRITS_imputation

#Read in the data 
logging.basicConfig(filename='CN_Missings.log',
                        level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("CN_Missings")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)
num_simulations = 50
best_validation_rmse = 1000000
logging.info("Performing hyperparameter optimization")
    
dataset = 'CN_daily_Final'
target, x_loc, y_loc = 'temp_top', 'longitude', 'latitude'
missing_percentage = 'masked'
df_ground_truth = load_ground_truth_data(dataset)
df_ground_truth = transform_data(df_ground_truth, 'time', target, x_loc, y_loc)
df_missings = load_missing_data(dataset, missing_percentage)
df_missings = transform_data(df_missings, 'time', target, x_loc, y_loc)
unique_comb = df_ground_truth[['longitude', 'latitude']].drop_duplicates()
method_functions = {
    'BRITS imputation': BRITS_imputation,
    'MRNN imputation': MRNN_imputation,
    'MIDA imputation': MIDA_imputation,
    'AKE imputation': AKE_imputation,
    'MICE imputation': mice_imputation,
    'KNN imputation': knn_imputer_function,
}
with open('utils/hyperparam_values.json', 'r') as file:
    hyperparam_data = json.load(file)

for method in [
    #'KNN imputation',
    #'MICE imputation', 
    #'AKE imputation',
    #'MIDA imputation',
    #'BRITS imputation',
    'MRNN imputation'
    ]:
    
    logging.info("Hyperparam optimization for method: {}".format(method))
    hyperparams = hyperparam_data.get(method, {})
    param_combinations = list(product(*hyperparams.values()))
    imputation_function = method_functions[method]
    for simulation in range(num_simulations):
        logging.info("Simulation {} out of {}".format(simulation + 1, num_simulations))
        X_train, X_test = train_test_split(unique_comb, test_size=0.03333, random_state=simulation)
        test_gt_df = df_ground_truth[df_ground_truth['longitude'].isin(X_test['longitude']) & 
                              df_ground_truth['latitude'].isin(X_test['latitude'])]
        test_ms_df = df_missings[df_missings['longitude'].isin(X_test['longitude']) & 
                              df_missings['latitude'].isin(X_test['latitude'])]
        random_param = random.choice(param_combinations)
        hyperparamer_sim = dict()
        for idx, key in enumerate(hyperparams.keys()):   
            hyperparamer_sim[key] = random_param[idx]
        if method in ['MIDA imputation','BRITS imputation','MRNN imputation']:
            df_imputed = imputation_function(test_ms_df,test_gt_df,df_gt=test_gt_df,**hyperparamer_sim)
        elif method in ['AKE imputation']:
            df_imputed = imputation_function(test_ms_df,test_gt_df,dataset='CN_daily_Final',**hyperparamer_sim)
        else:
            df_imputed = imputation_function(test_ms_df,test_gt_df,**hyperparamer_sim)
        rmse, mae, pce = simple_eval_masked("Simulation {} out of {}".format(simulation + 1, num_simulations),
                                            "{}".format(hyperparamer_sim),method, test_ms_df, df_imputed, test_gt_df)
        if (pce < 0):
            continue
        if (rmse < best_validation_rmse):
            logging.info("Updating best loss from {} to {}".format(
                        best_validation_rmse, rmse))
            best_validation_rmse = rmse
            best_hyperparams_sim = hyperparamer_sim.copy()    
    with open('utils/best_hyperparam_values.json', 'r') as file:
        hyperparam_data_final = json.load(file)
    hyperparam_data_final[method] = best_hyperparams_sim
    with open('utils/best_hyperparam_values.json', 'w') as file:
        json.dump(hyperparam_data_final, file, indent=2)

logging.info("End of hyperparam optimization!")
