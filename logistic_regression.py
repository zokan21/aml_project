import pandas as pd
import matplotlib.pyplot as plt 
import pandas as pd
import preprocessing
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score



data = pd.read_csv('dataR2.csv')
X, y = preprocessing.preprocessing()
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y)

lr = LogisticRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(lr.score(X_test, y_test))
print('#'*60)
print(classification_report(y_test, y_pred))

scores = cross_validate(LogisticRegression(), X_train, y_train, cv = 5, scoring=('roc_auc', 'average_precision'))
print('ROC AUC mean : {}'.format(scores['test_roc_auc'].mean()))
print('Average precision mean : {}'.format(scores['test_average_precision'].mean()))