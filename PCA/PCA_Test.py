from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from PCA import PCA
import pandas as pd

data = datasets.load_digits()
data = datasets.load_iris()
X = data.data
y = data.target

print(X.shape)

# 150, 4
# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print('Shape of X:', X.shape)
print('Shape of transformed X:', X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2,
            c=y, edgecolor='none', alpha=0.8,
            cmap=plt.cm.get_cmap('viridis', 3))
plt.colorbar()
#plt.show()

x1 = np.expand_dims(x1, axis=1)
x2 = np.expand_dims(x2,axis=1)

# print(np.append(x1, x2, axis=1))

print(y)
