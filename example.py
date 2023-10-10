import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis = 1).values
Y = df['Outcome'].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=0)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knn_score = knn.score(X_train, Y_train)
print(knn_score)

pickle.dump(knn, open('example_weights_knn.pkl', "wb")) 