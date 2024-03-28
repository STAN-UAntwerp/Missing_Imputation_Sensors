import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import UndefinedMetricWarning
from pathlib import Path
from utils.load_data import load_sensor_ids, load_sensor_distances


def AKE_imputation(dataframe,df_ground_truth, dataset, **kwargs):

    """
    :param dataframe: pandas dataframe with missing values
    :param k: number of nearest neighbors to consider
    :param dataset: dataset name

    :return: pandas dataframe with imputed values
    """
    k = kwargs.get('k', 5)

    def linreg(x_input, y_input):
        """
        :param x_input: pandas series
        :param y_input: pandas series
        :return: the linear regression predictor and the coefficient of determination
        """

        # reshape to fit the linreg requirements
        x = x_input.values.reshape(-1, 1)
        y = y_input.values.reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        # if there are less than two samples, r2 is ill-defined, and it will return a np.nan
        r2 = reg.score(x, y)
        return reg, r2
    
    def limit_sensors(pivotted_df, df_sensor_ids, df_sensor_distances):
        """
        :param pivotted_df: pandas dataframe with missing values
        :param df_sensor_ids: pandas dataframe with sensor ids
        :param df_sensor_distances: pandas dataframe with sensor distances
        :return: pandas dataframe with limited sensor ids and distances
        """
        # get the longitude and latitude combinations in the pivotted dataframe
        cols = pivotted_df.columns

        # find the corresponding sensor ids in the sensor ids dataframe
        df_sensor_ids = df_sensor_ids[df_sensor_ids.apply(lambda x: (x['longitude'], x['latitude']) in cols, axis=1)]

        # limit the rows of the distances
        df_sensor_distances = df_sensor_distances[df_sensor_distances.index.isin(df_sensor_ids.index)]

        # limit the columns of the distances
        df_sensor_distances = df_sensor_distances[df_sensor_distances.columns[df_sensor_distances.columns.astype(int).isin(df_sensor_ids.index)]]

        return df_sensor_ids, df_sensor_distances

    def get_k_nearest_neighbors(longitude, latitude, k, df_sensor_ids, df_sensor_distances):
        """
        :param longitude: longitude of the sensor
        :param latitude: latitude of the sensor
        :param k: number of nearest neighbors to consider
        :param df_sensor_ids: pandas dataframe with sensor ids
        :param df_sensor_distances: pandas dataframe with sensor distances
        :return: list of k nearest neighbors
        """

        # get the sensor id of the sensor with given longitude and latitude
        sensor_id = df_sensor_ids[(df_sensor_ids['longitude'] == longitude) & (df_sensor_ids['latitude'] == latitude)].index[0]
        # get the k nearest neighbors
        neighbors = df_sensor_distances[str(sensor_id)].nsmallest(k + 1).index.tolist()
        neighbors.remove(sensor_id)

        return neighbors
    
    def combine_neighbors(series, pivotted_df, neighbors, df_sensor_ids):
        """
        :param dataframe: pandas dataframe with missing values
        :param series: pandas series with missing values
        :param neighbors: list of k nearest neighbors
        :param df_sensor_ids: pandas dataframe with sensor ids
        :return: pandas series with the combined values of the neighbors
        """

        # create empty dataframe to collect predictions based on neighbor sensors
        predictions_df = pd.DataFrame(index=series[series.isna()].index, columns=neighbors)
        # empty dict to store the coefficients of determination
        r2_dict = {}

        for neighbor_id in neighbors:
            neighbor_longitude = df_sensor_ids.loc[neighbor_id, 'longitude']
            neighbor_latitude = df_sensor_ids.loc[neighbor_id, 'latitude']

            neighbor_series = pivotted_df.iloc[:, pivotted_df.columns.get_loc((neighbor_longitude, neighbor_latitude))]

            # fit the linreg

            x = neighbor_series[~neighbor_series.isna() & ~series.isna()]
            y = series[~neighbor_series.isna() & ~series.isna()]

            if not x.empty:
                reg, r2 = linreg(x, y)
                        
            # make the predictions for all NaNs in the series; if all values are NaN, all predictions will be NaN
            neigbor_data = neighbor_series[series.isna() & ~neighbor_series.isna()]
            # if there are no predictions, or the training data was empty (also no predictions), fill all predictions with NaN
            if neigbor_data.empty or x.empty:
                preds = [np.nan] * len(series[series.isna()])

            else:
                predictions = reg.predict(neigbor_data.values.reshape(-1, 1))
                # add NaNs for values that can not be predicted (because the neighbor has no value for that index)
                # this could probably be improved
                preds = []
                iterator = iter(predictions)
                for i in neighbor_series[series.isna()]:
                    if np.isnan(i):
                        preds.append(np.nan)
                    else:
                        # add [0] to avoid getting an array
                        preds.append(next(iterator)[0])

            # add the predictions to the dataframe and r2 to the dict
            predictions_df[neighbor_id] = preds
            r2_dict[neighbor_id] = r2 if not x.empty else np.nan

        # combine all predictions
        total_r2 = sum(r2_dict.values())

        # normalize the r2 values (this will be the weights for the predictions)
        weights = {k: v / total_r2 for k, v in r2_dict.items()}

        # in the predictions df, multiply each column with its weight (this works as dict keys are same as column names)
        predictions_df = predictions_df * weights

        # for every row, if there are NaNs, divide the row by the sum of the USED (non-NaN) weights
        # this is necessary because the weights are normalized over all neighbors, but not all neighbors are used
        columns = predictions_df.columns
        for row_id in predictions_df[predictions_df.isna().any(axis=1)].index:
            
            # check which columns are used for this row
            weights_used = ~predictions_df.loc[row_id].isna()
    
            total_weights_used = 0
            for column in columns:
                if weights_used[column]:
                    total_weights_used += weights[column]

            # if total_weights_used is 0, then all the weights are 0, so we can't do any imputation
            if total_weights_used:

                # divide the weights with the row values to fix the normalization
                predictions_df.loc[row_id] = predictions_df.loc[row_id] / total_weights_used
            
        # sum the predictions. We set min_count to 1, so that if there are no predictions, we will get a NaN
        predictions_df = predictions_df.sum(axis=1, numeric_only=True, min_count=1)

        # add the predictions to the series. If a prediction is still NaN, it will stay this way
        series[series.isna()] = predictions_df

        return series
    
    def AKE(series, dataframe, k, df_sensor_ids, df_sensor_distances):
        """
        :param series: pandas series with missing values
        :param dataframe: pandas dataframe with missing values
        :param k: number of nearest neighbors to consider
        :param df_sensor_ids: pandas dataframe with sensor ids
        :param df_sensor_distances: pandas dataframe with sensor distances
        :return: pandas series with imputed values
        """
        # get the longitude and latitude of the sensor
        longitude, latitude = series.name

        # get the ids of the k nearest neighbors
        neighbors = get_k_nearest_neighbors(longitude, latitude, k, df_sensor_ids, df_sensor_distances)

        # then we combine the neighbors and add the predictions to the series
        series = combine_neighbors(series, dataframe, neighbors, df_sensor_ids)

        return series

    # get the sensor ids and distances
    df_sensor_ids = load_sensor_ids(dataset)
    df_sensor_distances = load_sensor_distances(dataset)

    # pivot the dataframe
    pivotted_df = dataframe.pivot(index='time', columns=['longitude', 'latitude'], values='target')

    # limit the sensor ids and distances to the sensors in the dataframe
    df_sensor_ids, df_sensor_distances = limit_sensors(pivotted_df, df_sensor_ids, df_sensor_distances)

    # apply the AKE imputation
    imputed_df = pivotted_df.apply(lambda x: AKE(x, pivotted_df, k, df_sensor_ids, df_sensor_distances), axis=0)

    # unpivot the dataframe
    unpivoted_df = imputed_df.reset_index().melt(id_vars='time', var_name=['longitude', 'latitude'], value_name='target')
    
    return unpivoted_df