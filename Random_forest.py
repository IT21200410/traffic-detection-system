import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pickle
# import os
# cwd = os.getcwd()
# print(cwd)

df = pd.read_csv('traffic_processed.csv')

x = df.drop('Traffic_Level', axis = 1)
y = df['Traffic_Level']

from imblearn.combine import SMOTEENN
smote = SMOTEENN(random_state=42)
x_smoteen, y_smoteen = smote.fit_resample(x, y)
# from imblearn.over_sampling import SMOTE

# smote = SMOTE(random_state = 42)

# # fit predictor and target variable
# x_smote, y_smote = smote.fit_resample(x, y)

X_train, X_test, y_train, y_test = train_test_split(x_smoteen, y_smoteen, test_size = 0.2, shuffle = True)

random_forest = RandomForestClassifier(n_estimators=15, max_features = 'log2', max_depth = 15)
random_forest.fit(X_train, y_train)

pickle.dump(random_forest, open('Randommodel.pkl', 'wb'))