import pandas as pd
import seaborn as sns
import pickle
from imblearn.combine import SMOTEENN
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split

data = pd.read_csv('traffic_processed.csv')

X = data.drop('Traffic_Level', axis=1)
y = data['Traffic_Level']


smote = SMOTEENN(random_state=42)
x_smoteen, y_smoteen = smote.fit_resample(X, y)



X_train, X_test, y_train, y_test = train_test_split(x_smoteen, y_smoteen, test_size = 0.3)


 
classifier = GaussianNB()  
classifier.fit(X_train, y_train)


pickle.dump(classifier, open('NaiveBayes.pkl', 'wb'))