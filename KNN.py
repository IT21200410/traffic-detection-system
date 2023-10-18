from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import calendar
import pickle

df = pd.read_csv('traffic.csv')

def week_of_month(tgtdate):
    
    days_this_month = calendar.mdays[tgtdate.month]
    for i in range(1, days_this_month):
        d = datetime.datetime(tgtdate.year, tgtdate.month, i)
        if d.day - d.weekday() > 0:
            startdate = d
            break
    # now we canuse the modulo 7 appraoch
    return (tgtdate - startdate).days //7 + 1

def convertTraffic(data):

    if data <= 30:
        return 'Low'
    elif data <= 60:
        return 'Medium'
    else:
        return 'High'


df['DateTime'] = pd.to_datetime(df['DateTime'])
df['week'] = df['DateTime'].apply(week_of_month)
df['day'] = df['DateTime'].dt.day_name()
df['traffic_time'] = df['DateTime'].dt.time
df['month'] = df['DateTime'].dt.month
df['Traffic_Level'] = df.apply(lambda x: convertTraffic(x['Vehicles']), axis = 1) 
df['day'] = df['day'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
for i in range(len(df)):
    df['traffic_time'][i] = int(str(df['traffic_time'][i])[:2])


df.drop('DateTime', axis = 1, inplace = True)
df.drop('ID', axis = 1, inplace = True)
df.drop('Vehicles', axis = 1, inplace = True)


x = df.drop('Traffic_Level', axis = 1)
y = df['Traffic_Level']



smote = SMOTE(k_neighbors = 3, random_state = 100)
# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(x, y)

X_train, X_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size = 0.2, shuffle = True)
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train.ravel())


pickle.dump(classifier, open('KNN_model.pkl','wb'))
