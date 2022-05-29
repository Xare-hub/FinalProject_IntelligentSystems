from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from PCA import PCA
import pandas as pd

# Each column units are kg in a m^3 mixture
# Except the Age column, which is in days
# and the Compressive stregth column, which is in Megapascals

# Read data as a dataframe with Pandas
data = pd.read_excel(r"C:\Users\javie\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Universidad\10mo Semestre\SI\FinalProject\Datasets\Concrete\Concrete_Data.xls")

# Transform Pandas dataframe to numpy array
np_data = data.to_numpy()

# Define X and y
X = np_data[:, 0:7]
y = np_data[:, 8]


print(X.shape)
print("y shape is:", y.shape)

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
            edgecolor='none', alpha=0.8,
            cmap=plt.cm.get_cmap('viridis', 3))
plt.colorbar()
plt.show()

x1 = np.expand_dims(x1, axis=1)
x2 = np.expand_dims(x2,axis=1)

# print(np.append(x1, x2, axis=1))
