from cluster_validity_indices.metric import calculate_dunn_index
import numpy as np
from numpy import genfromtxt
from sklearn.cluster import KMeans, AgglomerativeClustering
from pbm_Index import cal_pbm_index
from sklearn.metrics import silhouette_score
from normalization import NormalizatorFactory

# from sklearn.metrics.cluster.supervised import adjusted_rand_score
# from minkowski_score import  cal_minkowski_score

Idata = genfromtxt('dataset/nips/nips_count_term_docvector_183d.csv', skip_header=0, usecols=range(0, 183))
print "Doc shape : ", Idata.shape


# print "data shape : ", Idata.shape
# normalizer = NormalizatorFactory.build('var')
# Idata=normalizer.normalize(Idata)


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


# Give no. of clusters

K = 20

# --------------------------------



cluster = KMeans(n_clusters=K)
cluster.fit(Idata)
label = cluster.predict(Idata)
centers = cluster.cluster_centers_
dn = calculate_dunn_index(Idata, label, K)
print "Kmeans Siloutte score : ", silhouette_score(Idata, label)
print "Kmeans PBM index : ", cal_pbm_index(K, Idata, centers, label)
print "Kmeans Dunn index : ", dn
print "-------------------------------------------------------------\n"

agglomerative1 = AgglomerativeClustering(n_clusters=K, linkage='ward', affinity='euclidean')
idx = agglomerative1.fit_predict(Idata)
hlabels = agglomerative1.labels_
centers = cal_center(Idata, hlabels)
dn = calculate_dunn_index(Idata, hlabels, K)
print "Hierarchical single Siloutte score : ", silhouette_score(Idata, hlabels)
print "Hierarchical single PBM index : ", cal_pbm_index(K, Idata, centers, hlabels)
print "Hierarchical single Dunn index : ", dn
print "---------------------------------------------------------\n"

agglomerative = AgglomerativeClustering(n_clusters=K, linkage='average', affinity='euclidean')
agglomerative.fit_predict(Idata)
hlabels = agglomerative.labels_
centers = cal_center(Idata, hlabels)
dn = calculate_dunn_index(Idata, hlabels, K)
print "Hierarchical average Siloutte score : ", silhouette_score(Idata, hlabels)
print "Hierarchical average PBM index : ", cal_pbm_index(K, Idata, centers, hlabels)
print "Hierarchical average Dunn index : ", dn
print "-------------------------------------------------------------\n"

agglomerative2 = AgglomerativeClustering(n_clusters=K, linkage='complete', affinity='euclidean')
agglomerative2.fit_predict(Idata)
hlabels = agglomerative2.labels_
centers = cal_center(Idata, hlabels)
dn = calculate_dunn_index(Idata, hlabels, K)
print "Hierarchical complete Siloutte score : ", silhouette_score(Idata, hlabels)
print "Hierarchical complete PBM index : ", cal_pbm_index(K, Idata, centers, hlabels)
print "Hierarchical complete Dunn index : ", dn
# print "hlabels : ", hlabels
print "---------------------------------------------------------\n"
