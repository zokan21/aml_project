import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree, export_graphviz
import pandas as pd

data = pd.read_csv('dataR2.csv')

y = data['Classification']
X = data.iloc[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y)

tree = DecisionTreeClassifier(max_depth=4, criterion='gini')
tree.fit(X_train, y_train)

plt.figure(figsize=(10, 3), dpi=100)
# tree_dot = plot_tree(tree, feature_names=data.columns)

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
param_grid = {'max_depth':range(1, 8)}
grid = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid=param_grid, cv=4, return_train_score=True)
grid.fit(X_train, y_train)

scores = pd.DataFrame(grid.cv_results_)

scores.plot(x='param_max_depth', y='mean_train_score', yerr='std_train_score', ax=plt.gca(), figsize=(20,5))
scores.plot(x='param_max_depth', y='mean_test_score', yerr='std_test_score', ax=plt.gca(), figsize=(20,5))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel('max_depth', fontsize=26)
plt.legend(fontsize=18)
plt.show()
