import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from cluster_validity_indices.pbm_Index import cal_pbm_index


def count_solution(y_copy, data, A1, z1, chromosome, sil_score, pbm_index, cluster, q, max_cluster, features):
    de = []
    fy = []
    count = 0

    cluster_center = np.reshape(y_copy, (
    -1, data.shape[1]))  ## Converts the generated solution y into the form of cluster number|features
    pp = len(cluster_center)
    cluster_new = KMeans(n_clusters=pp, init=cluster_center, max_iter=100, n_init=1)  ## Apply k-means
    cluster_new.fit(data)
    new_center = cluster_new.cluster_centers_  ## Find the new centres
    label = cluster_new.predict(data)  ## Find the new labels
    new_ss = silhouette_score(data, label)  ## Find the new silhouette score
    fy.insert(0, new_ss)  ##  Create a list of f(y)
    new_pbm = cal_pbm_index(cluster[q], data, new_center, label)  ## Find the new pbm index
    fy.insert(1, new_pbm)

    for k in range(0, chromosome):
        w1 = np.random.uniform(0, 1)
        w2 = 1 - w1
        lam1 = (1 / float(w1)) / ((1 / float(w1)) + (1 / float(w2)))
        lam2 = (1 / float(w2)) / ((1 / float(w1)) + (1 / float(w2)))
        m1 = lam1 * abs(fy[0] - z1[0])
        m2 = lam2 * abs(fy[1] - z1[1])
        gy = min(m1, m2)
        m1 = lam1 * abs(sil_score[k] - z1[0])
        m2 = lam2 * abs(pbm_index[k] - z1[1])
        gx = min(m1, m2)
        de.insert(k, (gy - gx))
    # print de
    # print sorted(de, reverse=True)
    x = [ii[0] for ii in sorted(enumerate(de), key=lambda x: x[1])]  ## Store the index of de in ascending order in x
    I = x[::-1]  ## Store the indices of de in descending order in I by reversing x
    # print A

    for jj in range(0, chromosome):
        if de[jj] < 0.1 and de[I[jj]] > -0.1:
            count = count + 1  ## Increase count by 1 if de(jj) > 0 for j = 0, 1, ... N
    # print A
    return count
