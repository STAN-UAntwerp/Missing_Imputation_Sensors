import pandas as pd
from pathlib import Path


def transform_time_IN(df):

    df['time'] = df['date'] + ' ' + df['time']
    df.drop(['date'], axis=1, inplace=True)
    # remove the decimal point and everything after it
    df['time'] = df['time'].apply(lambda x: x if '.' not in x else x[:x.find('.')])

    return df


def transform_data(dataframe,time,target,location_x,location_y):
    """
    Convert the 'time' column to numeric format (seconds since epoch) if it's in datetime format.
    Check if 'time' is already a number and if it is in ascending order for each longitude, latitude.

    Parameters:
    - dataframe: input dataframe, time column name, target column name, location column name coordinates x, 
    location column name coordinates y
    
    Returns:
    - dataframe with time column transformed 
    """
    if ':' in dataframe[time].iloc[0]:
        dataframe[time] = pd.to_datetime(dataframe[time], format="%Y-%m-%d %H:%M:%S").astype('int64') / 1e9

    else:
        dataframe[time] = pd.to_datetime(dataframe[time]).astype('int64') / 1e9

    if not pd.api.types.is_numeric_dtype(dataframe[time]):
        raise ValueError("'time' column is not numeric.")
    
    if not dataframe.groupby([location_x, location_y])[time].is_monotonic_increasing.all():
        # raise ValueError("'time' is not in ascending order for each 'id'.")
        # sort the dataframe by time
        dataframe = dataframe.sort_values(by=[time])
    
    column_name_mapping = {time: 'time', target: 'target',location_x:'longitude',location_y:'latitude'}
    dataframe.rename(columns=column_name_mapping, inplace=True)
    
    return dataframe

def load_missing_data(dataset: str, missing_amount: str) -> pd.DataFrame:

    """
    Load the missing data from the csv file.
    missing_amount can either be a percentage, or a string 'masked', for a realistic missings scenario.
    """

    datapath = Path(__file__).parent.parent.resolve() / 'data' / dataset
    df = pd.read_csv(f"{datapath}/{missing_amount}_missings.csv", dtype={'id': int, 'index': int, 'date': str, 'time': str, 'temp_top': float, 
                                                                         'longitude': float, 'latitude': float, 'moteid': int, 'temperature': float,
                                                                         'x_location': float, 'y_location': float})
    # if there is a date column (aka, it is the CN dataset), combine it with the time column
    if 'date' in df.columns:
        df = transform_time_IN(df)
    # drop sensor id, index and moteid column as it does not contain physical information
    for column in ['id', 'index', 'moteid']:
        if column in df.columns:
            df = df.drop(column, axis=1)
    # if there is a column that starts with 'Unnamed', drop it
    for column in df.columns:
        if 'Unnamed' in column:
            df = df.drop(column, axis=1)
    return df


def load_imputed_data(method: str, dataset: str, missing_amount: str) -> pd.DataFrame:

    """
    Loads the imputed data from the csv file.
    """
    results_path = Path(__file__).parent.parent.resolve() / 'results' / dataset / method
    df = pd.read_csv(f"{results_path}/{str(missing_amount)}_imputed.csv")
    # reorder columns if necessary
    df = df[['time', 'target', 'longitude', 'latitude']]
    return df


def load_ground_truth_data(dataset: str) -> pd.DataFrame:

    """
    Loads the ground truth data from the csv file.
    """

    datapath = Path(__file__).parent.parent.resolve() / 'data' / dataset
    df = pd.read_csv(f"{datapath}/ground_truth.csv", dtype={'id': int, 'index': int, 'date': str, 'time': str, 'temp_top': float, 
                                                            'longitude': float, 'latitude': float, 'moteid': int, 'temperature': float,
                                                            'x_location': float, 'y_location': float})
    # if there is a date column (aka, it is the CN dataset), combine it with the time column
    if 'date' in df.columns:
        df = transform_time_IN(df)
    # drop sensor id, index and moteid column as it does not contain physical information
    for column in ['id', 'index', 'moteid']:
        if column in df.columns:
            df = df.drop(column, axis=1)
    return df


def load_sensor_ids(dataset: str) -> pd.DataFrame:

    """
    Loads the sensor ids from the csv file.
    """

    datapath = Path(__file__).parent.parent.resolve() / 'data' / dataset
    df = pd.read_csv(f"{datapath}/sensor_ids.csv", dtype={'longitude': float, 'latitude': float}, index_col=0)
    return df


def load_sensor_distances(dataset: str) -> pd.DataFrame:

    """
    Loads the sensor distances from the csv file.
    """

    datapath = Path(__file__).parent.parent.resolve() / 'data' / dataset
    df = pd.read_csv(f"{datapath}/sensor_distances_hv.csv", index_col=0)
    return df