"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json







def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    df_all=feature_vector_df
    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    
    
    
    
            # functions
    def to_seconds(x):
            v = pd.DatetimeIndex(df_all[x])
            df_all[x+'_'+'seconds'] = v.hour*60*60+v.minute*60+v.second
            return df_all
    def convert_per_five_day(x):
            if x < 6:
                return 1
            if x < 11 and x >= 6:
                return 2
            if x < 16 and x >= 11:
                return 3
            if x < 21 and x >= 16:
                return 4
            if x < 26 and x >= 21:
                return 5
            return 6
    def grouped_features(Id, feature, new_name, df):
            group = df.groupby(Id).sum().reset_index()[[Id, feature]]
            group[new_name] = group[[feature]]
            group = group.drop(feature, axis=1)
            df = df.merge(group, how='left', on=Id)
            return df
    def calculate_distance(Latitude, Longitude, latitude2, longitude):
            R = 6373.0

            Latitude = np.radians(Latitude)
            Longitude = np.radians(Longitude)
            latitude2 = np.radians(latitude2)
            longitude = np.radians(longitude)

            dlon = longitude - Longitude
            dlat = latitude2 - Latitude

            x = np.sin(dlat / 2)**2
            y = np.cos(Latitude) * np.cos(latitude2)
            a = x + y * np.sin(dlon / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            return R * c


           
        
        
        
        
        
    
    
    
    
    df_all['Temperature'] = df_all.groupby('Placement - Day of Month')['Temperature'].transform(lambda x: x.fillna(x.median()))

    df_all.drop('Precipitation in millimeters', axis=1, inplace=True)


    

    #df_all.drop(['Arrival at Destination - Day of Month'], axis=1, inplace=True)

    df_all= to_seconds('Placement - Time')
    df_all = to_seconds('Confirmation - Time')
    df_all = to_seconds('Pickup - Time')
    df_all = to_seconds('Arrival at Pickup - Time')


    # Getting the difference in the times
    df_all['dif_confirm'] = df_all['Confirmation - Time_seconds'] - \
        df_all['Placement - Time_seconds']
    df_all['dif_arrival'] = df_all['Arrival at Pickup - Time_seconds'] - \
        df_all['Confirmation - Time_seconds']
    df_all['dif_pick'] = df_all['Pickup - Time_seconds'] - \
        df_all['Arrival at Pickup - Time_seconds']


    """Feature Engineering  ----------------------------------------------------------------------"""

    df_all['Ranking_order'] = df_all['No_of_Ratings'] / df_all['No_Of_Orders']
    df_all['Ranking_Distribution'] = (df_all['No_Of_Orders'] - df_all['Age']) / df_all['No_Of_Orders']

    """Feature Engineering Days ----------------------------------------------------------------------"""


            

    df_all['Per_Fiveday_Ofmonth'] = df_all['Placement - Day of Month'].apply(convert_per_five_day)


    """Feature Engineering Groupby----------------------------------------------------------------------"""
    # Group categories and perfom aggregates


    df_all = grouped_features('Rider Id', 'Distance (KM)','Total_distance_covered(KM)', df_all)
    df_all = grouped_features('Rider Id', 'dif_arrival', 'Tot_sec_picup_dest(KM)', df_all)
    df_all['riders_speed'] = df_all['Total_distance_covered(KM)'] / df_all['Tot_sec_picup_dest(KM)']

    """Feature Engineering Geospacial----------------------------------------------------------------------"""

# Converting the coordinates
# Can also be done using geopy package



    df_all['calc_distance'] = calculate_distance(df_all['Pickup Lat'], df_all['Pickup Long'], df_all['Destination Lat'], df_all['Destination Long'])

    df_all['Change_Distance'] = df_all['Distance (KM)'] - df_all.calc_distance
    df_all['Pick_Dist'] = df_all['Pickup Lat'] + df_all['Pickup Long']
    df_all['Destination_Dist'] = df_all['Destination Lat'] + df_all['Destination Long']
    df_all['orderedtransformed'] = df_all.No_Of_Orders**(1/5.1)
    df_all['averageRating'] = df_all.Average_Rating**(1/0.5)
    df_all['age_tranformed'] = df_all.Age**(1/4)
    df_all['distanceinkmtransformed'] = df_all['Distance (KM)']**(1/1.9)
    df_all['ampmconfirm'] = df_all['Confirmation - Time'].astype('str').apply(lambda x: x.split(' ')[-1])


    """Dropping Data----------------------------------------------------------------------"""


    df_all = df_all.drop(['Confirmation - Day of Month', 'Arrival at Pickup - Day of Month',
                          'Confirmation - Weekday (Mo = 1)', 'Arrival at Pickup - Weekday (Mo = 1)',
                          'Placement - Time_seconds', 'Confirmation - Time_seconds',
                          'Arrival at Pickup - Time_seconds', 'Pickup - Time_seconds',
                          'No_of_Ratings', 'calc_distance', 'Placement - Time',
                          'Confirmation - Time', 'Pickup - Time', 'Arrival at Pickup - Time',
                          'Total_distance_covered(KM)', 'Tot_sec_picup_dest(KM)', 'riders_speed',
                          'ampmconfirm', 'Order No', 'User Id', 'Vehicle Type', 'Rider Id','Arrival at Destination - Time'], axis=1)

#
# The code below is for demonstration purposes only. You will not
# receive marks for submitting this code in an unchanged state.
# ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = df_all[['Age', 'Average_Rating', 'Destination Lat', 'Destination Long',
           'Distance (KM)', 'No_Of_Orders', 'Pickup - Day of Month',
           'Pickup - Weekday (Mo = 1)', 'Pickup Lat', 'Pickup Long',
           'Placement - Day of Month', 'Placement - Weekday (Mo = 1)',
           'Platform Type', 'Temperature',
           'dif_confirm', 'dif_arrival', 'dif_pick', 'Ranking_order',
           'Ranking_Distribution', 'Per_Fiveday_Ofmonth',
           'Change_Distance', 'Pick_Dist', 'Destination_Dist',
           'orderedtransformed', 'averageRating', 'age_tranformed',
           'distanceinkmtransformed', 'Personal or Business_Business',
           'Personal or Business_Personal']]
    # ------------------------------------------------------------------------
    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
