import numpy as np
import matplotlib.pyplot as plt
import utils
import kmeans

def main():
    X = utils.generateClusters(10000, 3)
    utils.plotDatapoints(X)    

    classifier = kmeans.KMeans(3)
    classifier.fit(X)
    classifier.plotSegmentation(X)
    return 0

if __name__ == "__main__":
    main()