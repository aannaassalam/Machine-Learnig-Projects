import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples = 15, n_features = 2, centers = 3, cluster_std = 1, 
                 shuffle = True)

print(X)

plt.scatter(X[:,0], X[:,1], c = 'white', marker = 'o', edgecolor = 'black', s =30)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Datapoints')
plt.show()

from scipy.cluster.hierarchy import linkage, dendrogram

link = linkage(X, method= 'single')

plt.figure(figsize = (10,7))

dendrogram(link)

plt.title("Dendrogram of Hierarchical Clustering")
plt.ylabel("Distance")

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')

y_ac = ac.fit_predict(X)

plt.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], s = 30, c = 'cyan', marker = 's', edgecolor = 'black', 
            label = "Cluster-1")

plt.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], s = 30, c = 'green', marker = 'o', edgecolor = 'black',
            label = "Cluster-2")

plt.scatter(X[y_ac == 2, 0], X[y_ac == 2, 1], s = 30, c = 'blue', marker = 'v', edgecolor = 'black', 
            label = "Cluster-3")

plt.legend()
plt.show()