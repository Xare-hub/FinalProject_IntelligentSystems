import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # calculate mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # calculate covariance matrix
        # x is np.ndarray where 1 row = 1 sample and 1 column = 1 feature
        # Checking the documentation, in numpy this is the other way around
        cov = np.cov(X.T)
        # calculate eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # v[:,1] <--- Each column is an eigenvector
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)
