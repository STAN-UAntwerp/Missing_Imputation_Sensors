#General import
import logging
import pandas as pd
from pathlib import Path
import warnings
from utils.run_data import data_run
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    logging.basicConfig(filename='CN_Missings.log',
                        level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("CN_Missings")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.info("Start script!")

    # Make results dataframe
    results_df = pd.DataFrame(columns=['Missings','Method', 'RMSE', 'MAE', 'PCE','Execution time'])  

    # Run all methods 
    #results_df_out = data_run(results_df,'masked',True,False,True)

    #for missing_perc in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    #    results_df_out_ms = data_run(results_df,str(missing_perc),True,True,True)
    #    results_df_out= pd.concat([results_df_out,results_df_out_ms])

    #results_df_out.to_csv('results/CN_daily_Final/ALL_CN_daily_results.csv', index=False)    
    #results_df = pd.read_csv('results/CN_daily_Final/ALL_CN_daily_results.csv')
    
    results_df_out_2 = data_run(results_df,'masked',True,False,False)
    results_df_out= pd.concat([results_df,results_df_out_2])

    #results_df_out.to_csv('results/CN_daily_Final/ALL_CN_daily_results.csv', index=False)

    #results_df_out = pd.read_csv('results/CN_daily_Final/ALL_CN_daily_results.csv')
    
    for missing_perc in [0.1,0.2,0.3,0.4,0.5]:
        results_df_out_ms = data_run(results_df_out,str(missing_perc),True,True,False)
        results_df_out = pd.concat([results_df_out,results_df_out_ms])

    results_df_out.to_csv('results/CN_daily_Final/ALL_CN_daily_results2.csv', index=False)
    logger.info("End script!")
