import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import neighbors
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import ensemble 
import scipy

#importing the dataset
rawdf = pd.read_csv("creditcard.csv", delimiter = ",")

PCA_list = rawdf.columns[1:29] #contains the labels V1, V2, ..., V28
PCA_index = np.arange(1,29)

df = rawdf.copy()
cols_to_drop = ["V6", "V8", "V13", "V15", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
               "Amount", "Time", "V18", "V16", "V7", "V5"]
df.drop(cols_to_drop, inplace=True, axis=1)

Y = df.Class
X = df.drop("Class", axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                stratify=Y, 
                                                test_size=0.3) #training/test 70/30 splitting

#undersampling
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler()
X_under, Y_under = undersampler.fit_sample(X_train, np.ravel(Y_train))

#parameters to be used with the random_forest method
max_depth =3 
max_features = 3
n_estimators = 260


rf = ensemble.RandomForestClassifier(max_depth=max_depth,
									 max_features=max_features, 
									 n_estimators=n_estimators) #classifier for original data
rf_under = ensemble.RandomForestClassifier(max_depth=max_depth,
									 max_features=max_features, 
									 n_estimators=n_estimators) #classifier for undersampled

rf.fit(X_train, np.ravel(Y_train))

rf_under.fit(X_under, np.ravel(Y_under))

Y_pred = rf.predict(X_test)
Y_under_pred = rf_under.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(Y_test,Y_pred))

print(classification_report(Y_test,Y_under_pred))




