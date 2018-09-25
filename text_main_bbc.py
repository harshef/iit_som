import math
import random
# from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from numpy import genfromtxt, linalg, apply_along_axis
from cluster_validity_indices.metric import cal_pbm_index, silhouette_score, calculate_dunn_index
# from cluster_validity_indices.minkowski import cal_minkowski_score
from clustering.clusternew import Kmeans_clu
from evolution.generate import reproduction
from mapping.matingpool import mating_pool
from mapping.somnew import mapping, no_of_neurons, neuron_weights_init
from population_creation.population import pop_create
from update.update_sol import update_solution, calculatez
from update.probability import probability
from update.count import count_solution

file = open('Text_Outputs/BBC1/BC_gensim/run1/bbc_gensin_50d_output_run1.txt', 'w')
file1= open('Text_Outputs/BBC1/BC_gensim/run1/bbc_gensin_50d_output_generation_details_run1.txt', 'w')

data = genfromtxt('dataset/BBC1/BBC/BBC_gensim/BBC_gensim_50d.csv', skip_header=0, usecols=range(0, 50)) ##Read the input data
actual_label = genfromtxt('dataset/BBC1/BBC/BBC_gensim/BBC_gensim_50d.csv', skip_header=0, usecols=(50))
chromosome = int(input("Enter the number of chromosomes: "))  # Input the population size
max_gen = int(input("Enter the maximum number of generation: "))  # Input the maximum number of generation
max_cluster = math.floor(math.sqrt(len(data)))  ##Maximum cluster equals root(population)
features = len(data[0])  ##Features of the input data
max_cluster_feat = max_cluster * features  ##Maximum cluster features possible
K = []  # record no. of clusters of solutions in population
lab = []  # record predicted labels of clusters of solutions in population
center = []
population = []
sil_score = []
ini_dunn = []
pbm_index = []
tau_init = 0.8
CN = {}
RN = {}
for mm in range(0, max_gen):  ## Initialize dictionary of size upto length maximum generation
    CN[mm] = {}
    RN[mm] = {}

for i in range(0, chromosome):
    cluster = random.randint(2, max_cluster)  ##Randomly selects cluster number from 2 to root(poplation)
    K.insert(i, cluster)  ##Store the number of clusters in clu
    u, label = Kmeans_clu(cluster, data)
    lab.insert(i, label)  ##Store the labels in lab
    center.insert(i, u)
    new_pbm = cal_pbm_index(cluster, data, u, label)
    pbm_index.insert(i, new_pbm)  ##Store the second objective function(PBM Index)
    dn = calculate_dunn_index(data, label, cluster)
    ini_dunn.insert(i, dn)
    s = silhouette_score(data, label)
    sil_score.insert(i, s)  ##Store the first objective funvtion(Silhouette Score)
    new_center = pop_create(max_cluster, features, cluster, u)
    population.insert(i, new_center)  ##Store the populations in population

file.write("Initial Clusters : " + str(K) + '\n')
file.write("Initial PBM : " + str(pbm_index) + '\n')
file.write("Initial Sil score : " + str(sil_score) + '\n')
file.write("Initial Dunn : " + str(ini_dunn) + '\n')
index = ini_dunn.index(max(ini_dunn))
file.write("Max Dunn and corresponding cluster, PBM, Sil : " + str(ini_dunn[index]) + "," + str(K[index]) + "," + str(
    pbm_index[index]) + "," + str(sil_score[index]) + '\n')
file.write("------------------------------------------------------------------------------------------\n")

z = calculatez(sil_score, pbm_index)  ## Z* initialization
# Intializations
A = population[:]  ## Create an archive A of population
A_K = K[:]
C, D = no_of_neurons(chromosome)
neu_weights, neu_weights_k = neuron_weights_init(population, D, K)
S = population[:]
S_K = K[:]
v_pos = [4, 6, 8, 10]  ##size of different NNS
beta = []  # NNS probability
for i in range(len(v_pos)):
    beta.insert(i, 1 / float(len(v_pos)))

cn = np.zeros(len(v_pos))
rn = np.zeros(len(v_pos))
for t in range(0, max_gen):

    neu_weights, L, neu_weights_k = mapping(S, chromosome, population, tau_init, t, max_gen, C, D, neu_weights,
                                            neu_weights_k, S_K, K, features)
    # from collections import Counter
    # a = Counter(L)
    # idx = np.random.choice(chromosome, len(v_pos))
    idx = random.sample(range(0, chromosome), len(v_pos))
    idx = np.asarray(idx)
    A1 = A[:]  ##Step 7 of algorithm 1 that is set A' = A
    z1 = z[:]  ##Step 7 of algorithm 1 that is set z' = z
    lab_copy = lab[:]  ## Copy of list of labels for A
    sil_score_copy = sil_score[:]  ## Copy of list of objective function1(silhouette score) for A
    pbm_index_copy = pbm_index[:]  ## Copy of list of objective function2(pbm index) fo.

    # r A
    K_copy = K[:]  ## Copy of list of clusters for A

    for q in range(0, chromosome):
        if np.any(idx == q):
            a = np.nonzero(idx == q)  ##Check if idx contains q
            v_nns = v_pos[a[0][0]]  ##Select the NNS size from v_pos based on the index on which idx contains q
            pos = a[0][0]  ##Assign the index where q matches in idx to pos
            # print v_nns
        else:
            aa = max(enumerate(beta), key=lambda x: x[1])[0]  ##Find the index of maximum probability from the list beta
            pos = aa  ##Assign the index of maximum probability in the list 'beta' to pos
            v_nns = v_pos[aa]  ##Select the NNS size from v_pos corresponding to the maximum probability
        m_pool, B_K = mating_pool(q, L, K, v_nns, chromosome, population)
        y = reproduction(population, q, m_pool, K, features, data, max_cluster)
        A, z, K_copy, lab_copy, sil_score_copy, pbm_index_copy = update_solution(y, data, A, chromosome, sil_score,
                                                                                 pbm_index, K, q, max_cluster, features,
                                                                                 lab_copy, sil_score_copy,
                                                                                 pbm_index_copy, K_copy)
        count = count_solution(y, data, A1, z1, chromosome, sil_score, pbm_index, K, q, max_cluster, features)
        cn[pos] = cn[pos] + 1
        rn[pos] = rn[pos] + count

    for uu in range(len(v_pos)):
        CN[t][uu] = cn[uu]
        RN[t][uu] = rn[uu]
    beta = probability(t, len(v_pos), CN, RN, max_gen)
    ##<!!---Update the training set---!!>
    S = []
    S_K = []
    uu = 0
    for tt in range(0, chromosome):
        if np.array_equal(A[tt], population[tt]):
            continue
        else:
            S.insert(uu, A[tt])
            S_K.insert(uu, K_copy[tt])
            uu += 1
    population = A[:]
    K = K_copy[:]
    lab = lab_copy[:]
    sil_score = sil_score_copy[:]
    pbm_index = pbm_index_copy[:]
    file1.write("Generation no  :  " + str(t) + '\n')
    file1.write("Clusters : " + str(K) + '\n')
    file1.write("Predicted labels : " + str(lab) + '\n')
    file1.write("PBM index : " + str(pbm_index) + '\n')
    file1.write("Sil score : " + str(sil_score) + '\n')
    file.write( "-------------------------------------------------------------------------------------------------------------------------\n\n")
    print "--------------------generation {0} completed----------------".format(t)

final_dunn = []
for i in range(chromosome):
    file.write("chromosome {0} : ".format(i) + str(population[i]))
    file.write("\n Objectives (Sil) : " + str(sil_score[i]))
    file.write("\n Objectives (PBM) : " + str(pbm_index[i]))
    file.write("\nCluster size {0} :".format(i) + str(K[i]) + '\n')
    file.write("Predicted Label for {0}: ".format(i) + str(lab[i]))
    dunn = calculate_dunn_index(data, lab[i], K[i])  # Pass the data, predicted label and number of clusters
    final_dunn.insert(i, dunn)
    file.write("\n Dunn Index {0} : ".format(i) + str(dunn))
    # ari = adjusted_rand_score(actual_label, lab[i])
    # final_ARI.insert(i, ari)
    # mink = cal_minkowski_score(actual_label, lab[i])
    # final_MS.insert(i, mink)
    # file.write("\n ARI Score {0} : ".format(i)+ str(ari))
    # file.write("\nMinkowski score {0} : ".format(i)+str(cal_minkowski_score(actual_label, lab[i])))
    file.write(
        "\n---------------------------------------------------------------------------------------------------------\n")

print "Max Dunn Index : ", max(final_dunn)
index = final_dunn.index(max(final_dunn))
print "No of clusters detected automatically : ", K[index]
# print "Max ARI : ",max(final_ARI)
print "PBM and Sil score coressponding to maximum Dunn : ", pbm_index[index],",", sil_score[index]

file.write(
    "\n-------------------------------------------------------------------------------------------------------------\n")
file.write(
    "\n-------------------------------------------------------------------------------------------------------------\n")
file.write("\n Maximum dunn_index for {0}th chromosome : ".format(index) + str(final_dunn[index]) + '\n')
# file.write("\n Minkowski score corresponding to maximum ARI : " + str(final_MS[index])+'\n')
file.write("\n Maximum silhouette score : " + str(max(sil_score)) + '\n')
file.write("\n Maximum pbm index : " + str(max(pbm_index)) + '\n')
file.write("Final label corresponding to max Dunn for {0} chromosome : ".format(index) + str(lab[index]) + '\n')
# file.write("Actual label of data : " + str(actual_label))
file.write("\n No of clusters detected automatically : " + str(K[index]))
file.write("\n For chromosome : " + str(index))

import matplotlib.pyplot as plt

plt.plot(sil_score, pbm_index, 'bo')
# plt.grid(True)
plt.grid(True, color='0.75', linestyle='-')
plt.xlabel('Silhouette score')
plt.ylabel('PBM index')
# plt.axis([0, max(return_ss), 0, max(return_pbm)])
plt.savefig('Text_Outputs/BBC1/BC_gensim/run1/bbc_gensin_50d_run1.jpeg')
plt.show()
