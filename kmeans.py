import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k : int, maxiter : int = 1000) -> None:
        self.k = k
        self.maxiter = maxiter

    def __init_centroids(self, X : pd.DataFrame, k : int) -> None:
        x = X['x'].values
        y = X['y'].values

        random_indices = np.random.choice(X.shape[0], size = k, replace = False)
        self.centroids = np.column_stack((x[random_indices], y[random_indices]))
        
    def __init_centroids_pp(self, X : pd.DataFrame, k : int) -> None:
        """TODO: Fix computation of euclidean distances"""
        points : np.ndarray = X.values
        x : np.ndarray = X['x'].values
        y : np.ndarray = X['y'].values
        n : int = X.shape[0]

        centroids : list = []
        # Choose the first centroid randomly from the data points
        index : int = np.random.choice(n, size = 1, replace = False)
        centroids.append((x[index], y[index]))

        # Choose the remaining K-1 centroids
        for iter in range(k - 1):
            distances = np.zeros(n)
            for idx in range(n):
                for center in range(len(centroids)):
                    distances[idx] = self.L2Norm(points[idx], centroids[center])
            probabilities : np.ndarray = (distances / distances.sum())
            cumulative_probabilities : np.ndarray = probabilities.cumsum()
            r : int = np.random.rand()
            next_centroid = points[cumulative_probabilities >= r][0]
            centroids.append(next_centroid)
        print(centroids)

        self.centroids = centroids
        self.centroids = pd.DataFrame(self.centroids, columns=['x', 'y'])
        self.centroids = self.centroids.values

    def __assign_clusters(self, X : pd.DataFrame, k : int, n : int) -> None:
        points = X.values
        labels = np.zeros(n)

        for i in range(n):
            distances = np.zeros(k)
            for j in range(k):
                distances[j] = self.L2Norm(points[i], self.centroids[j])
            labels[i] = np.argmin(distances)

        self.labels = labels

    def __update_centroids(self, X : pd.DataFrame, k : int) -> None:
        for i in range(k):
            indices = (self.labels == i)
            points = X[indices]
            self.centroids[i] = points.mean()
    
    def plotSegmentation(self, X : pd.DataFrame):
        X = X.drop('cluster', axis = 1)
        X = X.values

        # Create a color map for the clusters
        colors = plt.cm.Spectral(np.linspace(0, 1, self.k))

        # Plot the data points
        for i in range(self.k):
            mask = (self.labels == i)
            plt.scatter(X[mask, 0], X[mask, 1], color=colors[i], s=30)

        # Plot the centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', color='k', s=100)

        # Set the axis labels
        plt.xlabel('x')
        plt.ylabel('y')

        # Show the plot
        plt.show()

    def fit(self, X : pd.DataFrame) -> None:
        X = X.drop('cluster', axis = 1)
        self.__init_centroids(X, self.k)
        self.__assign_clusters(X, self.k, X.shape[0])

        iter = 0
        while iter != self.maxiter:
            self.__update_centroids(X, self.k)
            iter += 1

    def L2Norm(self, a1 : np.ndarray, a2 : np.ndarray):
        return np.sqrt(((a1 - a2)**2).sum())

