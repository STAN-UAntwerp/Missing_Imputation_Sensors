from scipy.spatial.distance import cdist
from geopy.distance import geodesic
import pandas as pd

def haversine(coord1, coord2):
    return geodesic(coord1, coord2).km


def calculate_closest_points(df_unique,method="euclidean"):
    """
    Calculates the distance between two points/sensors

    Parameters:
    - dataframe: input dataframe with the unique columns (longitude, latitude)
    - method: how to calculate the distance (haversine or euclidean)

    Returns:
    - dataframe of format X by X that has the distance for each sensor similar to a correlation matrix
    """

    if method == "haversine":
        distances = pd.DataFrame(
            cdist(df_unique[['latitude', 'longitude']], df_unique[['latitude', 'longitude']], metric=haversine),
            index=df_unique.index,
            columns=df_unique.index)
    elif method == "euclidean":
        distances = pd.DataFrame(
            cdist(df_unique[['latitude', 'longitude']], df_unique[['latitude', 'longitude']], metric='euclidean'),
            index=df_unique.index,
            columns=df_unique.index)
    else:
        raise ValueError("Invalid method. Supported methods: 'haversine', 'euclidean'.")

    return distances