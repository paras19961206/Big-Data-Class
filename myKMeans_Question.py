
"""
K-means clustering (Question)
"""
import numpy as np

def myKMeans(dataset, k = 3, seed = 42):
    """Returns the centroids of k clusters for the input dataset.
    
    Parameters
    ----------
    dataset: a pandas DataFrame
    k: number of clusters
    seed: a random state to make the randomness deterministic
    
    Examples
    ----------
    myKMeans(df, 5, 123)
    myKMeans(df)
    
    Notes
    ----------
    The centroids are returned as a new pandas DataFrame with k rows.
    
    """
    np.random.seed(seed)
    dataset_f = getNumFeatures(dataset)
    old_centers = getInitialCentroids(dataset_f,k,seed)
    new_centers = computeCentroids(dataset_f, getLabels(dataset_f,old_centers))
    iterations = 1
    while not stopClustering(old_centers, new_centers, iterations):
        old_centers = new_centers
        new_centers = computeCentroids(dataset_f, getLabels(dataset_f,old_centers))
        iterations += 1
    return new_centers 

def getNumFeatures(dataset):
    """Returns a dataset with only numerical columns in the original dataset"""
    return dataset._get_numeric_data()

def getInitialCentroids(dataset, k, seed):
    """Returns k randomly selected initial centroids of the dataset"""
    np.random.seed(seed)
    mean = np.mean(dataset, axis = 0)
    std = np.std(dataset, axis = 0)
    centers = np.random.randn(k,len(dataset.columns))*std.tolist() + mean.tolist()     
    return centers

def euc(row, centroid):
    r = row.tolist()
    distance = 0
    for i in range(len(r)):
        distance += (centroid[i] - r[i])**2
    return np.sqrt(distance)

def getLabels(dataset, centroids):
    """Assigns labels (i.e. 0 to k-1) to individual instances in the dataset.
    Each instance is assigned to its nearest centroid.
    """ 
    distances = np.zeros((len(dataset), len(centroids)))
    for i in range(len(centroids)):
        for ind,row in dataset.iterrows():
            distances[ind][i] = euc(row, centroids[i])
    labels = np.zeros(len(dataset))
    i = 0
    for r in distances:
        min_dist = r.argmin()
        labels[i] = min_dist
        i += 1
    return labels

def computeCentroids(dataset, labels):
    """Returns the centroids of individual groups, defined by labels, in the dataset"""
    dataset['labels'] = labels
    aver = dataset.groupby('labels').mean().values.tolist()
    dataset.drop('labels', axis=1, inplace=True)
    return aver


def stopClustering(oldCentroids, newCentroids, numIterations, maxNumIterations = 100, tol = 1e-4):
    """Returns a boolean value determining whether the k-means clustering converged.
    Two stopping criteria: 
    (1) The distance between the old and new centroids is within tolerance OR
    (2) The maximum number of iterations is reached 
    """
    net_dist = 0
    for i in range(len(oldCentroids)):
        for j in range(len(oldCentroids[0])):
            net_dist += (oldCentroids[i][j] - newCentroids[i][j])**2
    return np.sqrt(net_dist) <= tol or maxNumIterations <= numIterations
    
    
        