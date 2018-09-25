from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np


def Kmeans_clu(K, data):
    kmeans = KMeans(n_clusters=K, init='random', max_iter=1, n_init=1).fit(data)  ##Apply k-means clustering
    labels = kmeans.labels_
    clu_centres = kmeans.cluster_centers_
    return clu_centres, labels


def average_hierarchical_cluster(K, data):
    agglomerative = AgglomerativeClustering(n_clusters=K, linkage='average', affinity='euclidean')
    agglomerative.fit_predict(data)
    hlabels = agglomerative.labels_
    centers = cal_center(data, hlabels)
    u = centers.values()
    return u, hlabels


def cal_center(data, T):
    """
    :param features: data points
    :param T: Cluster label of data points
    :return:
    """
    lens = {}  # will contain the lengths for each cluster
    centroids = {}  # will contain the centroids of each cluster
    for idx, clno in enumerate(T):
        centroids.setdefault(clno, np.zeros(len(data[0])))
        centroids[clno] += data[idx, :]
        lens.setdefault(clno, 0)
        lens[clno] += 1
    # Divide by number of observations in each cluster to get the centroid
    for clno in centroids:
        centroids[clno] /= float(lens[clno])
    return centroids
