import numpy as np

def pop_create(max_cluster, features, cluster, center):
    max_cluster_feat = max_cluster * features   ##Maximum cluster features possible
    z = int(max_cluster_feat - (cluster * features))  ##Calculate number of zeros to append(Maximum cluster dimension - Current cluster dimension)
    center = np.reshape(center, cluster*features)
    for j in range(z):
        center = np.append(center, [0])
    return center