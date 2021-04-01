import numpy as np
import matplotlib.pyplot as plt 
import preprocessing
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import plot_roc_curve


data = pd.read_csv('dataR2.csv')
X, y = preprocessing.preprocessing()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

rf_reg = RandomForestRegressor( random_state = 42 )

# Finding the optimal number of trees parameter

estimator_range = range(10, 310, 10)

rmse_scores = []

for estimator in estimator_range:
    rf_reg = RandomForestRegressor( n_estimators = estimator, random_state = 42 )
    mse_scores = cross_val_score(rf_reg, X_train, y_train, scoring = 'neg_mean_squared_error', n_jobs = 1, cv = 5)
    rmse_scores.append(np.mean(np.sqrt(-mse_scores)))

plt.plot(estimator_range, rmse_scores)
plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.show()

L = []

for k in range(len(estimator_range)):
    L.append((rmse_scores[k], estimator_range[k],))
    
L_sorted = sorted(L)
err, num_est = L_sorted[0]
print('Minimum error is {} and its corresponding number of estimators is {}.'.format(err, num_est))

# The result is : n_estimators = 120

# Optimizing max_feature parameter

feature_range = range(1, data.shape[1])
rmse_scores_feat = []

for feature in feature_range:
    rf_reg = RandomForestRegressor(n_estimators=num_est, max_features=feature, random_state=42)
    mse_scores_feat = cross_val_score(rf_reg, X_train, y_train, scoring = 'neg_mean_squared_error', n_jobs = 1, cv = 5)
    rmse_scores_feat.append(np.mean(np.sqrt(-mse_scores_feat)))

plt.plot(feature_range, rmse_scores_feat)
plt.xlabel('max_features')
plt.ylabel('RMSE')
plt.show()

L = []

for k in range(len(feature_range)):
    L.append((rmse_scores_feat[k], feature_range[k],))
    
L_sorted = sorted(L)
err_feat, max_feat = L_sorted[0]
print('Minimum error is {} and its corresponding maximum number of features is {}.'.format(err_feat, max_feat))

# The result is : max_features = 2

# Training the classifier with the optimum parameters

rf_reg = RandomForestRegressor(n_estimators = num_est, max_features = max_feat, random_state = 42)
rf_reg.fit(X_train, y_train)
train_score = rf_reg.score(X_train, y_train)
test_score = rf_reg.score(X_test, y_test)

print('Train score is {}% and test score is {}%.'.format(train_score*100, test_score*100))

# Plotting feature importance

feat_imp = rf_reg.feature_importances_
print(feat_imp)
print(data.columns)

# We conclude that Age, Glucose, BMI and Resistin are the most important variables to predict breast cancer

# ROC curves

rf_roc = plot_roc_curve(rf_reg, X_test, y_test, alpha = 0.8 )
plt.show()