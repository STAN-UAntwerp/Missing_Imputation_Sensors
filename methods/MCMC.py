import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.enable_eager_execution()
 
def MCMC_imputation(df_missings,df_ground_truth):
    
    # Pivot missing data
    pivot_df_ms = df_missings.pivot(index='time', columns=['longitude', 'latitude'], values='target')

    # Impute all sensors simultaneously
    time_series_with_nans = pivot_df_ms.values
    is_missing = tf.math.is_nan(time_series_with_nans)
    observed_time_series = tfp.sts.MaskedTimeSeries(
        time_series=time_series_with_nans,
        is_missing=is_missing)

    # Build model using observed time series to set heuristic priors
    linear_trend_model = tfp.sts.LocalLinearTrend(
        observed_time_series=observed_time_series)
    
    model = tfp.sts.Sum([linear_trend_model],
                       observed_time_series=observed_time_series)

    # Fit model to data
    parameter_samples, _ = tfp.sts.fit_with_hmc(model, observed_time_series)

    # Impute missing values
    observations_dist = tfp.sts.impute_missing_values(
        model, observed_time_series, parameter_samples).sample()
    
    # Replace missing values in the original DataFrame
    pivot_df_ms.loc[:, :] = np.where(is_missing, observations_dist.numpy(), time_series_with_nans)

    unpivoted_df = pivot_df_ms.reset_index().melt(id_vars='time', var_name=['longitude', 'latitude'], value_name='target')

    return unpivoted_df