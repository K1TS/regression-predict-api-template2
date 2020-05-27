"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
import numpy as np
from math import sin, cos, sqrt, atan2
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling


def read_data(data_file_list):
    return [pd.read_csv(f) for f in data_file_list]


data_frame_list = read_data(['train_data.csv', 'test_data.csv', 'Riders.csv'])

train = data_frame_list[0]
test = data_frame_list[1]
riders = data_frame_list[2]



# Fusing delivery and rider datasets
train = train.merge(riders, how='left', on='Rider Id')
test = test.merge(riders, how='left', on='Rider Id')

# Combining all training + test data for feature engineering purposes
df_all = pd.concat([train, test], sort=True)

# Model changes
""""Droping and Replacing--------------------------------------------------------------- """

df_all['Temperature'] = df_all.groupby(
    'Placement - Day of Month')['Temperature'].transform(lambda x: x.fillna(x.median()))

df_all.drop('Precipitation in millimeters', axis=1, inplace=True)

df_all.drop(['Arrival at Destination - Weekday (Mo = 1)'],
            axis=1, inplace=True)

df_all.drop(['Arrival at Destination - Day of Month'], axis=1, inplace=True)

""""Time Convertion ----------------------------------------------------------------------"""

# Converting time format to seconds
def to_seconds(x):
    v = pd.DatetimeIndex(df_all[x])
    df_all[x+'_'+'seconds'] = v.hour*60*60+v.minute*60+v.second
    return df_all


df_all = to_seconds('Placement - Time')
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


""""Feature Engineering  ----------------------------------------------------------------------"""

df_all['Ranking_order'] = df_all['No_of_Ratings'] / df_all['No_Of_Orders']
df_all['Ranking_Distribution'] = (
df_all['No_Of_Orders'] - df_all['Age']) / df_all['No_Of_Orders']

""""Feature Engineering Days ----------------------------------------------------------------------"""


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


df_all['Per_Fiveday_Ofmonth'] = df_all['Placement - Day of Month'].apply(
    convert_per_five_day)


""""Feature Engineering Groupby----------------------------------------------------------------------"""
# Group categories and perfom aggregates
def grouped_features(Id, feature, new_name, df):
    group = df.groupby(Id).sum().reset_index()[[Id, feature]]
    group[new_name] = group[[feature]]
    group = group.drop(feature, axis=1)
    df = df.merge(group, how='left', on=Id)
    return df


df_all = grouped_features('Rider Id', 'Distance (KM)','Total_distance_covered(KM)', df_all)
df_all = grouped_features('Rider Id', 'dif_arrival', 'Tot_sec_picup_dest(KM)', df_all)
df_all['riders_speed'] = df_all['Total_distance_covered(KM)'] / df_all['Tot_sec_picup_dest(KM)']

""""Feature Engineering Geospacial----------------------------------------------------------------------"""

# Converting the coordinates
# Can also be done using geopy package
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


df_all['calc_distance'] = calculate_distance(
    df_all['Pickup Lat'], df_all['Pickup Long'], df_all['Destination Lat'], df_all['Destination Long'])






df_all['Change_Distance'] = df_all['Distance (KM)'] - df_all.calc_distance
df_all['Pick_Dist'] = df_all['Pickup Lat'] + df_all['Pickup Long']
df_all['Destination_Dist'] = df_all['Destination Lat'] + df_all['Destination Long']
df_all['orderedtransformed'] = df_all.No_Of_Orders**(1/5.1)
df_all['averageRating'] = df_all.Average_Rating**(1/0.5)
df_all['age_tranformed'] = df_all.Age**(1/4)
df_all['distanceinkmtransformed'] = df_all['Distance (KM)']**(1/1.9)
df_all['ampmconfirm'] = df_all['Confirmation - Time'].astype('str').apply(lambda x: x.split(' ')[-1])


""""Dropping Data----------------------------------------------------------------------"""


df_all = df_all.drop(['Confirmation - Day of Month', 'Arrival at Pickup - Day of Month',
                      'Confirmation - Weekday (Mo = 1)', 'Arrival at Pickup - Weekday (Mo = 1)',
                      'Placement - Time_seconds', 'Confirmation - Time_seconds',
                      'Arrival at Pickup - Time_seconds', 'Pickup - Time_seconds',
                      'No_of_Ratings', 'calc_distance', 'Placement - Time',
                      'Confirmation - Time', 'Pickup - Time', 'Arrival at Pickup - Time',
                      'Total_distance_covered(KM)', 'Tot_sec_picup_dest(KM)', 'riders_speed',
                      'ampmconfirm', 'Order No', 'User Id', 'Vehicle Type', 'Rider Id','Arrival at Destination - Time'], axis=1)




y_train = df_all[:len(train)][['Time from Pickup to Arrival']]
df_all = pd.get_dummies(df_all)
train = df_all[:len(train)].drop('Time from Pickup to Arrival', axis=1)
test = df_all[len(train):].drop('Time from Pickup to Arrival', axis=1)



y_train = y_train[['Time from Pickup to Arrival']]
X_train = train[['Age', 'Average_Rating', 'Destination Lat', 'Destination Long',
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

# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = 'sendy1.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))
