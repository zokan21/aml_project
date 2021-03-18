import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy
import pandas as pd 

data = pd.read_csv('dataR2.csv')
print(data[data['Classification'] == 1].describe())
print(data[data['Classification'] == 2].describe())

healthy = data[data['Classification'] == 1]
cancer = data[data['Classification'] == 2]

fig, axs = plt.subplots(3, 3)

axs[0,0].boxplot([healthy['Age'], cancer['Age']])
axs[0,0].set_title('Age')

axs[0,1].boxplot([healthy['BMI'], cancer['BMI']])
axs[0,1].set_title('BMI')

axs[0,2].boxplot([healthy['Glucose'], cancer['Glucose']])
axs[0,2].set_title('Glucose')

axs[1,0].boxplot([healthy['Insulin'], cancer['Insulin']])
axs[1,0].set_title('Insulin')

axs[1,1].boxplot([healthy['HOMA'], cancer['HOMA']])
axs[1,1].set_title('HOMA')

axs[1,2].boxplot([healthy['Leptin'], cancer['Leptin']])
axs[1,2].set_title('Leptin')

axs[2,0].boxplot([healthy['Adiponectin'], cancer['Adiponectin']])
axs[2,0].set_title('Adiponectin')

axs[2,1].boxplot([healthy['Resistin'], cancer['Resistin']])
axs[2,1].set_title('Resistin')

axs[2,2].boxplot([healthy['MCP.1'], cancer['MCP.1']])
axs[2,2].set_title('MCP.1')

plt.show()

correlations = data.corr()
print(correlations)