import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

# X, y = datasets.make_regression(n_samples=9568, n_features=4, noise=1000, random_state=4)
# print(type(X[0,0]))
# print(type(y[0]))

# Read the data
data = pd.read_excel(r"C:\Users\javie\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Universidad\10mo Semestre\SI\FinalProject\Datasets\CCPP\Folds5x2_pp.xlsx")
np_data = np.array(data)
np_dataX = np_data[:, :-1]
np_datay = np_data[:, -1]

#Normalize data (Without normalization step, linear regression returns nan values due to vanishing/exploding gradients)
# Subtract out the mean
mean = np.mean(np_dataX, axis=0)
np_dataX = np_dataX - mean
# Normalize variance
var = 1/len(np_dataX) * np.sum(np_dataX**2)
std = np.sqrt(var)
np_dataX = np_dataX/std
# Add dimension to y vector to be able to append it no the normalized np_data array
np_datay = np.expand_dims(np_datay, axis=1)
# Append X and y
np_data = np.append(np_dataX, np_datay, axis=1)
print("\nnp_data shape: ", np_data.shape)
# Separate features and labels
X = np_data[:, :-1]
y = np_data[:, -1]
# Print sample instance
print("\nX[0]: ", X[0], "     y[0]: ", y[0], "\n")

#Separate dataset into training and tesing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# print(X_train.shape)
# print(y_train.shape)

# Import LinearRegression class
from LinearRegression import LinearRegression, mse

# Instantiate a LinearRegression object
regressor = LinearRegression()

# Fit the data to the linear regression model
regressor.fit(X_train, y_train)

# Predict values from the test set
predicted = regressor.predict(X_test)

# Compute mean squared error
mse_value = mse(y_test, predicted)
print("Mean squared error: ", mse_value, "\n")

# Print weights and bias vectors
print("Weights:", regressor.weights, "\nbias:",regressor.bias, "\n")

# Perform predictions
y_predictions = regressor.predict(X_test)

# Expand dimensions of y predictions to print as a column vector, purely aesthetical
print(np.expand_dims(y_predictions, axis=1))

# Define hyperparameters
exit = 0
features = []

Predict = input("\nDo you want to make a prediction of your own? (y/n)")
# Predict = 'Y'
while Predict == 'y':
    AT = input("\nEnter the Ambient Temperature")
    V = input("\nEnter the Exhaust Vacuum")
    AP = input("\nEnter the Ambient Pressure")
    RH = input("\nEnter the Relative Humidity")

    features = np.expand_dims(np.array([AT, V, AP, RH], dtype = 'float64'), axis=0)
    # Normalize inputs
    features = (features - mean)/std
    print(features)
    # Perform prediction
    new_prediction = regressor.predict(features)
    print("The expected power plant energy output is: ", new_prediction)

    Predict = input("Make another prediction? (y/n)")

print(features)
