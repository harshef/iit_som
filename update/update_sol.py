import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cluster_validity_indices.pbm_Index import cal_pbm_index


def calculatez(sil_score, pbm_index):
    z = []
    maximum = np.max(sil_score)  ## To initialise z find the max of objective function 1
    z.insert(0, maximum)
    maximum = np.max(pbm_index)  ## To initialise z find the max of objective function 2 as z = max|fx(j)|, j = 1, .. m
    z.insert(1, maximum)
    return z


def update_solution(y, data, A, chromosome, sil_score, pbm_index, cluster, q, max_cluster, features, lab_copy,
                    sil_score_copy, pbm_index_copy, K_copy):
    """
    To update the existing solutions with the new generated y using objective value of sub-problem
    :param y: The solution generated after repair and mutation
    :param data: The input data
    :param population: Input weight vector(population)
    :param chromosome: The number of population
    :param sil_score: The silhouette score generated for each population in a list
    :param pbm_index: The PBM index generated for each population in a list
    :param cluster: The cluster number of each population stored in a list
    :param q: The current solution number
    :return: Returns the updated solution in a archive point A, and the count of the number of solutions in A
             which could be replaced by the new solution y
    """

    fy = []
    de = []  ## Initialise a list of improvable degree de

    z = calculatez(sil_score, pbm_index)

    cluster_center = np.reshape(y, (
    -1, data.shape[1]))  ## Converts the generated solution y into the form of cluster number|features
    pp = len(cluster_center)
    K_copy[q] = pp  ## Update the new cluster in the copy of cluster list for A
    cluster_new = KMeans(n_clusters=pp, init=cluster_center, max_iter=100, n_init=1)  ## Apply k-means
    cluster_new.fit(data)
    # K_copy[q] = cluster[q]  ## Update the new cluster in the copy of cluster list for A
    new_center = cluster_new.cluster_centers_  ## Find the new centres
    label = cluster_new.predict(data)  ## Find the new labels
    lab_copy[q] = label  ## Update the new labels in the copy of labels list for A
    new_ss = silhouette_score(data, label)  ## Find the new silhouette score
    sil_score_copy[q] = new_ss  ## Update the new silhouette score in the copy of silhouette score list for A
    fy.insert(0, new_ss)  ##  Create a list of f(y)
    new_pbm = cal_pbm_index(cluster[q], data, new_center, label)  ## Find the new pbm index
    pbm_index_copy[q] = new_pbm  ## Update the new pbm index generated in the copy of pbm index list for A
    fy.insert(1, new_pbm)
    ##<!-----Update z* as if z*(j) < fy(j), for j =1, 2, ... m-----!>
    if z[0] < new_ss:
        z[0] = new_ss
    if z[1] < new_pbm:
        z[1] = new_pbm

    # print z
    ##<!-----Calculate objective value of subproblem i for y and x(i)-----!>##
    for k in range(0, chromosome):
        w1 = np.random.uniform(0, 1)
        w2 = 1 - w1
        lam1 = (1 / float(w1)) / ((1 / float(w1)) + (1 / float(w2)))
        lam2 = (1 / float(w2)) / ((1 / float(w1)) + (1 / float(w2)))
        m1 = lam1 * abs(fy[0] - z[0])
        m2 = lam2 * abs(fy[1] - z[1])
        gy = min(m1, m2)
        m1 = lam1 * abs(sil_score_copy[k] - z[0])
        m2 = lam2 * abs(pbm_index_copy[k] - z[1])
        gx = min(m1, m2)
        de.insert(k, (gy - gx))
    # print de
    # print sorted(de, reverse=True)
    x = [ii[0] for ii in sorted(enumerate(de), key=lambda x: x[1])]  ## Store the index of de in ascending order in x
    I = x[::-1]  ## Store the indices of de in descending order in I by reversing x
    # print A
    yy = y
    max_cluster_feat = max_cluster * features  ##Maximum cluster features possible
    z_zero = int(max_cluster_feat - (
    pp * features))  ##Calculate number of zeros to append(Maximum cluster dimension - Current cluster dimension)
    for j in range(z_zero):
        yy = np.append(yy, [0])
    for jj in range(0, chromosome):
        if jj == 0 or jj == 1:
            if de[I[jj]] < 0.1 and de[I[jj]] > -0.1:
                A[I[jj]] = yy  ## Replace x(I(jj) by y if de(I(jj))>0 for jj =1, 2
    # print A
    return A, z, K_copy, lab_copy, sil_score_copy, pbm_index_copy
