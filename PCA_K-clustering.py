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

# Define X and y for PCA algorithm
X = np_data
# y = np_data[:, 8]

# Print shapes
print("X shape is:", X.shape)
#print("y shape is:", y.shape)

# Reduce dimensionality
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print('Shape of X:', X.shape)
print('Shape of transformed X:', X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

# plt.scatter(x1, x2,
#             edgecolor='none', alpha=0.8,
#             cmap=plt.cm.get_cmap('viridis', 3))
# plt.colorbar()
# plt.show()

# Expand dimensions of principal component vectors to concatenate them and
# pass as data to the KMeans algorithm
x1 = np.expand_dims(x1, axis=1)
x2 = np.expand_dims(x2,axis=1)

X = (np.append(x1, x2, axis=1))

# set random seed for consistent results across tests
# np.random.seed(45)

from KMeansClustering import KMeans, euclidean_distance

print(type(X))
print(X.shape)
print(X[0])


k = KMeans(K=4, max_iters=150, plot_steps=True)
y_pred = k.predict(X)
print(y_pred)
