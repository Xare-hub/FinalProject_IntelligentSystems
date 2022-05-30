# DBSCAN portion of the code retrieved from: https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd
from PCA import PCA


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

# Expand dimensions of principal component vectors to concatenate them and
# pass as data to the KMeans algorithm
x1 = np.expand_dims(x1, axis=1)
x2 = np.expand_dims(x2,axis=1)
X = (np.append(x1, x2, axis=1))

# Compute DBSCAN

# Useful hyperparameters: [[30,15], [30,25]]
db = DBSCAN(eps=30, min_samples=25).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

print(db.labels_[:10])   # Prints labels for each of the groups
