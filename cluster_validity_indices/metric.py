from sklearn.metrics import silhouette_score
from cluster_validity_indices.pbm_Index import cal_pbm_index
from cluster_validity_indices.dunn_index import calculate_dunn_index


class cluster_indices():
    def __init__(self):
        return

    def pbm_index(self, cluster, data, center, label):
        return cal_pbm_index(cluster, data, center, label)

    def sil_score(self, data, label):
        return silhouette_score(data, label)

    def dunn_index(self, data, label, cluster):
        return calculate_dunn_index(data, label, cluster)
