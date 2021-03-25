import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def __main__():
    data = pd.read_csv('dataR2.csv')

    pipe = Pipeline([("Scaler", StandardScaler())]) 
    # No missing values nor categorical variables, so no more preprocessing to do !

    X = data.iloc[:,:-1]

    preprocessed_data = pipe.fit_transform(X)
    return preprocessed_data

