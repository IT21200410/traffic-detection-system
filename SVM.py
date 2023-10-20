import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('traffic_processed.csv')


target_column = 'Traffic_Level'
x = df.drop(columns=[target_column])
y = df[target_column]

smenn = SMOTEENN()
x_smote, y_smote = smenn.fit_resample(x, y)

X_train, X_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size = 0.2, shuffle = True)

classifier_svm = SVC(kernel='rbf', C=1.0)
classifier_svm.fit(X_train, y_train)

pickle.dump(classifier_svm, open('SVM_model.pkl','wb'))