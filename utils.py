import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 
import pandas as pd

def generateClusters(n, k):
    """Generate a dataset of n observations separated into k clusters"""
    X, y = make_blobs(n_samples=n, centers=k, cluster_std=0.8, random_state=42)
    X = pd.DataFrame(X, columns=['x','y'])
    X['cluster'] = y
    return X

def plotDatapoints(X):
    plt.scatter(X['x'], X['y'], c = X['cluster'])
    plt.show()
