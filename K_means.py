import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs  # For generating globular datapoints

X,y = make_blobs(n_samples = 200, n_features = 2, centers = 3, cluster_std = 1, 
                 shuffle = True)
print(X.shape)

plt.scatter(X[:,0], X[:,1], c = 'white', marker = 'o', edgecolor = 'black', s =20)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Datapoints')
plt.show()

def KMeans_Cluster(X, K, tol = 0.001):
    
    n_points = X.shape[0]
    
    assert K <= n_points, "Number of Clusters can't be greater than no. of points"
    
    # Choosing K initial centroids
    index = np.random.choice(n_points, K, replace = False)
    centroids = X[index]
    
    cluster_labels = np.empty(n_points)  # Place-holder for labels of each datapoints
    new_centroids = centroids.copy()     # Place-holder for new recomputed centroids
    flag = True
    
    while(flag):
        
        # Reassigning the data-points to different clusters
        for i in range(n_points):
            dist = np.linalg.norm(X[i]-centroids, axis = 1)
            cluster_labels[i] = np.argmin(dist)

        # Recompute the cluster centroids
        for j in range(K):
            new_centroids[j] = np.mean(X[cluster_labels == j], axis = 0)

        # Check the tolerence
        for k in range(K):
            if np.linalg.norm(new_centroids[k]-centroids[k]) > tol:
                flag = True
                break
            else:    
                flag = False
        
        centroids = new_centroids.copy()

    return new_centroids, cluster_labels

no_of_cluster = 3

centroids, cluster_labels = KMeans_Cluster(X, K = no_of_cluster, tol = 0.001)

print(centroids)

cols = ['r', 'g', 'b', 'c', 'm', 'y']
markers = ['o', 'v', 's', 'o', 'v', 's']

plt.figure(figsize=(7,7))

for j in range(no_of_cluster):
    x_plot = X[cluster_labels == j][:,0]
    y_plot = X[cluster_labels == j][:,1]
    plt.scatter(x_plot, y_plot,c = cols[j], s = 30, edgecolor = 'k', 
                marker = markers[j], label = 'cluster-'+str(j))

plt.scatter(centroids[:,0],centroids[:,1], c = 'c', s = 200, marker = '*', edgecolor = 'k',
           label = 'centroids')
plt.legend(loc = 'best')
plt.show()

from sklearn.cluster import KMeans  # From sklearn.cluster importing KMeans

km = KMeans(n_clusters = 3)

y_km = km.fit_predict(X)

plt.figure(figsize=(7,7))

plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s = 30, c = 'cyan', marker = 's', edgecolor = 'black', 
            label = "Cluster-1")

plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s = 30, c = 'green', marker = 'o', edgecolor = 'black',
            label = "Cluster-2")

plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s = 30, c = 'blue', marker = 'v', edgecolor = 'black', 
            label = "Cluster-3")

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s = 80, c = 'red', marker = '*', 
            label = "centers")

plt.legend()
plt.show()

print(km.cluster_centers_)

print(centroids)