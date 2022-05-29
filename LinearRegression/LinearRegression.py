# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

# d1=pd.read_csv(r"Student/student-mat.csv",sep=";")
# d1 = pd.read_csv("Wholesale_customers_data.csv")

class LinearRegression:

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):    # Involve training step and gradient descent
        # initialize parameters for GD
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Approximation
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        pass


    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted  # Makes predictions for unseen samples
