import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def preprocessing():
    data = pd.read_csv('dataR2.csv')

    pipe = Pipeline([("Scaler", StandardScaler())]) 
    # No missing values nor categorical variables, so no more preprocessing to do !

    X = np.array(data.iloc[:,:-1])
    y = np.array(data['Classification'].apply(lambda x : x - 1))

    preprocessed_X = pipe.fit_transform(X)
    return preprocessed_X, y

