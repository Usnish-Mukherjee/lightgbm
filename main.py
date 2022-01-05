import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import sklearn.metrics

startTime = datetime.now()
data = pd.read_csv(r"C:\AION\DataSet\PMLB\adult.csv")
print(data.head())
data.info()
## counting the output values
print(data['target'].value_counts())

X = data.iloc[:,:-1]
y = data.iloc[:,-1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)
y_pred_train = clf.predict(X_train)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

# view confusion-matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred_train)
print('Confusion matrix for training data\n', cm)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix for testing data\n', cm)


# # visualize confusion matrix with seaborn heatmap
#
# cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
#                                  index=['Predict Positive:1', 'Predict Negative:0'])
#
# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#classification report for training data
print("\t\tClassification report for training data\n\n")
from sklearn.metrics import classification_report
print(classification_report(y_train, y_pred_train))
print(sklearn.metrics.roc_auc_score(y_train, y_pred_train))

#classification report for testing data
print("\t\tClassification report for testing data\n\n")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print(sklearn.metrics.roc_auc_score(y_test, y_pred))

#Python 3:
print(datetime.now() - startTime)