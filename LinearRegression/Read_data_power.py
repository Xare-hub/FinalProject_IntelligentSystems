import pandas as pd
import numpy as np


data = pd.read_excel(r"C:\Users\javie\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Universidad\10mo Semestre\SI\FinalProject\Datasets\CCPP\Folds5x2_pp.xlsx")

data.info()


np_data = np.array(data)
np_dataX = np_data[:, :-1]
np_datay = np_data[:, -1]

print(np_dataX.shape)
print(np_datay.shape)

#Normalize data
# Subtract out the mean
mean = np.mean(np_dataX, axis=0)
np_dataX = np_dataX - mean
# Normalize variance
var = 1/len(np_dataX) * np.sum(np_dataX**2)
std = np.sqrt(var)
np_dataX = np_dataX/std

np_datay = np.expand_dims(np_datay, axis=1)

np_data = np.append(np_dataX, np_datay, axis=1)
print(np_data.shape)

X = np_data[:, :-1]
y = np_data[:, -1]
print(np_data)
print(X.shape)
print(y.shape)

# print(data.isnull().sum())
