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

X = (np.append(x1, x2, axis=1))

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(45)

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # List of sample indices for each cluster. We just store the indices
        self.clusters = [[] for _ in range(self.K)]
        # Store mean feature vector for each cluster. We have actual
        # samples
        self.centroids = []

    # Since we don't have labels, we don't need a fit method, just a predict

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K,
                                              replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimization
        for _ in range(self.max_iters):
            # Update clusters
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            # Update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self.plot_steps:
                self.plot()
            # Check if converged
            if self._is_converged(centroids_old, self.centroids):
                break

        # Return cluster labels
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):

        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def _create_clusters(self, centroids):


        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)

        return clusters

    def _closest_centroid(self, sample, centroids):
        """Calculate the distances of the current sample to each centroid.
           And then gets the centroid with the closest distance"""

        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)

        return closest_idx

    def _get_centroids(self, clusters):
        """Gets the centroids of the current cluster"""

        centroids = np.zeros((self.K,self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean

        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i])
                    for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        #%matplotlib inline
        fig, ax = plt.subplots(figsize=(12,8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()


import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#X, y = make_blobs(centers=5, n_samples=500, n_features=2, shuffle=True, random_state=42)
# print(X.shape)
# X = X.tolist()

print(type(X))
print(X.shape)
print(X[0])

#X_only = [sample[0] for sample in X]
#Y_only = [sample[1] for sample in X]

# print(X_only)
# print(Y_only)

#plt.scatter(X_only, Y_only)
#plt.show()

#clusters = len(np.unique(y))   #Prints number of clusters
#print(clusters)


k = KMeans(K=4, max_iters=150, plot_steps=True)
y_pred = k.predict(X)
#print(y_pred)
