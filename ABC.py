from cluster_validity_indices.minkowski import cal_minkowski_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from numpy import genfromtxt, linalg, apply_along_axis

data = genfromtxt('dataset/smile1.csv', delimiter=',', skip_header=1, usecols=range(0, 2)) ##Read the input data
actual_label = genfromtxt('dataset/smile1.csv', delimiter=',', skip_header=1, usecols=(2))
#data = apply_along_axis(lambda x: x/linalg.norm(x),1,data) # data normalization


kmeans = KMeans(n_clusters=4).fit(data) ##Apply k-means clustering
labels = kmeans.labels_
clu_centres = kmeans.cluster_centers_


print adjusted_rand_score(actual_label, labels)
print cal_minkowski_score(actual_label, labels)
