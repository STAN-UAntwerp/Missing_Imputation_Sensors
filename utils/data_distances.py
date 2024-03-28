#Import of necessary packages 
from distance_calc import calculate_closest_points
from load_data import load_missing_data, transform_data
    
def main():
     
    dataset, target, x_loc, y_loc, missing_percentage = 'CN_daily_Final', 'temp_top', 'longitude', 'latitude', 'masked'    
    df_missings = load_missing_data(dataset, missing_percentage)
    df_missings = transform_data(df_missings, 'time', target, x_loc, y_loc)
    unique_comb = df_missings[['latitude', 'longitude']].drop_duplicates()

    #Calculate the distances in euclidean 
    df_distances = calculate_closest_points(unique_comb,'euclidean')
    df_distances.columns = df_distances.columns.to_list()
    df_distances.to_csv('data/CN_daily_Final/sensor_distances_eu.csv', index=True)

    #Calculate haversine distances and store them
    df_distances = calculate_closest_points(unique_comb,'haversine')
    df_distances.columns = df_distances.columns.to_list()
    df_distances.to_csv('data/CN_daily_Final/sensor_distances_hv.csv', index=True)


def make_sensor_ids_df():

    dataset, target, x_loc, y_loc, missing_percentage = 'CN_daily_Final', 'temp_top', 'longitude', 'latitude', 'masked'    
    df_missings = load_missing_data(dataset, missing_percentage)
    df_missings = transform_data(df_missings, 'time', target, x_loc, y_loc)
    unique_comb = df_missings[['latitude', 'longitude']].drop_duplicates()
    unique_comb.to_csv('data/CN_daily_Final/sensor_ids.csv', index=True)
    #unique_comb.to_csv('data/CN_daily/sensor_ids.csv', index=True)
    
                                                         
if __name__ == "__main__":
    print("Calculate distances of sensors")
    main()
    make_sensor_ids_df()