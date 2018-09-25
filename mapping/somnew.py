import numpy as np
import math


def neuron_weights_init(pop1, D, K):
    """
        Initializes the weights of neurons from population.
        pop1 - input weight vector(population)
    """
    neu_weights = []  ##Weight of neurons
    neu_weights_k = []
    for ii in range(0, D):
        neu_weights.insert(ii, pop1[ii])  # Initialize neuron weights from populations
        neu_weights_k.insert(ii, K[ii])

    neu_weights = np.asarray(neu_weights)
    # print neu_weights
    # print("\n\n\n\n\n")
    return neu_weights, neu_weights_k


def position(minimum, C):
    """
    Calculates the position vector of the neuron
    :param minimum: Index of the winning neuron
    :return: Returns the position vector as x1, x2
    """
    x1 = int(minimum / C)  ##x-coordinate of the neuron
    x2 = int(minimum % C)  ##y-coordinate for the neuron
    return x1, x2


def eucledian_dist(x1, x2, k1, k2):
    """
    Calculates the eucledian distance between two neurons
    :param x1: x1 co-ordinate of the neuron 1
    :param x2: x2 co-ordinate of the neuron 1
    :param k1: k1 co-ordinate of the neuron 2
    :param k2: k2 co-ordinate of the neuron 2
    :return: returns the eucledian distance
    """
    return math.sqrt(((x1 - k1) ** 2) + (x2 - k2) ** 2)  # Distance between neuron(x1,x2) and (k1,k2)


def neuron_weights_update(S, chromosome, tau_init, D, C, t, T, neu_weights, neu_weights_k, K, features):
    """
    :param S: TRaining data for SOM
    :param chromosome: Number of solutions in population
    :param pop1: input weight vector(solutions in the population)
    :param tau: learning_rate - initial learning rate
        (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
        sigma - spread of the neighborhood function,
        (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
        decay_function, function that reduces learning_rate and sigma at each iteration
    :param D: Number of neurons for the given population
    :param C: no. of neurons in one row of 2D SOM  (equals floor value of square root(population))
    :param t: Current Generation Number
    :param T: Maximum Generation Number
    :return: Return the updated weight of the neuron
    """
    sig_init = C / math.sqrt(2)  ##Sigma initialization
    distance_train = []

    for iii in range(0, len(S)):
        xx = 1 - float(((t - 1) * chromosome) + iii) / float(chromosome * T)
        sig = sig_init * xx  # Update Sigma
        tau = tau_init * xx  # Update tau
        neu_weights_k = np.asarray(neu_weights_k)
        for jjj in range(0, D):
            if K[iii] < neu_weights_k[jjj]:
                main_cluster = K[iii]
            else:
                main_cluster = neu_weights_k[jjj]
            main_feat = main_cluster * features
            temp1 = S[iii]
            temp2 = neu_weights[jjj]
            a = temp1[:main_feat]
            b = temp2[:main_feat]
            dist = np.linalg.norm(a - b)
            distance_train.insert(jjj, dist)  ##List of distance between the current solution and all the neuron
        minimum_train = distance_train.index(min(distance_train))  # Index of the winning neuron for training
        # print distance_train
        # print minimum_train
        distance_train = []  ##Empty the list
        x1, x2 = position(minimum_train, C)  ##x,y-coordinate of the winning neuron for train
        for kkk in range(0, D):
            k1, k2 = position(kkk, C)  # x,y-cordinate for the kkkth neighboring neuron
            M = eucledian_dist(x1, x2, k1, k2)  # Distance of winning neuron and kkkth neuron
            if (M < sig):
                # print neu_weights[kkk]
                if K[iii] < neu_weights_k[jjj]:
                    main_cluster = K[iii]
                else:
                    main_cluster = neu_weights_k[jjj]
                main_feat = main_cluster * features
                temp3 = neu_weights[kkk]
                temp_neu_weights = temp3[:]
                temp1 = S[iii]
                a = temp1[:main_feat]
                b = temp3[:main_feat]
                temp_neu_weights[:main_feat] += tau * math.exp(-M) * (a - b)  ##Update weights of the neighboring neuron
                neu_weights = list(neu_weights)
                temp5 = neu_weights[kkk]
                temp5[:main_feat] = temp_neu_weights[:main_feat]
                # neu_weights_k[kkk] = main_cluster
                neu_weights = np.asarray(neu_weights)
                # print neu_weights[kkk]
    # print neu_weights
    # print ("\n")
    return neu_weights, neu_weights_k


def build_mapping(chromosome, D, pop1, neu_weights_update, neu_weights_k, K, features):
    """
    :param neu_weights_update: Updated weight of the neurons
    :return: Returns the Mapping of the population with neurons
    """
    distance_test = []
    L = []  ##Mapping Variable
    ##<!----------Create distinct neuron mapping for each solution, So a loop equal to the number of neurons is used---------!>##
    ##<!----------This ensures all the neurons are atlist mapped to one solution--------!>##
    for v in range(0, D):
        for vvv in range(0, D):
            if K[v] < neu_weights_k[vvv]:
                main_cluster = K[v]
            else:
                main_cluster = neu_weights_k[vvv]
            main_feat = main_cluster * features
            temp1 = pop1[v]
            temp2 = neu_weights_update[vvv]
            a = temp1[:main_feat]
            b = temp2[:main_feat]
            dist = np.linalg.norm(a - b)
            distance_test.insert(vvv, dist)
        # print distance_test
        ##minimum_test = distance_test.index(min(distance_test))  ##Index of winning neuron for test
        index = [i[0] for i in sorted(enumerate(distance_test), key=lambda x: x[
            1])]  ##Create a list of index contatining the index of distance_test in ascending order
        # print minimum_test
        for tt in range(0, len(distance_test)):
            if index[tt] in L:  ##If the minium distance is already present in L, thesn skip
                continue
            else:
                L.insert(v, index[
                    tt])  ##Else, build mapping of the winning neuron by inserting the next closest neuron that is now present in L
                neu_weights_k[index[tt]] = K[v]
                break  ##Once insertion in L is performed exit the loop
        distance_test = []  ##Empty the distance_test list
        index = []  ##Empty the index list
    distance_test = []
    ##<!--------Creating mapping for the leftover solutions to the neurons-----------!>##
    for gg in range(D, chromosome):
        for vvv in range(0, D):
            if K[gg] < neu_weights_k[vvv]:
                main_cluster = K[gg]
            else:
                main_cluster = neu_weights_k[vvv]
            main_feat = main_cluster * features
            temp1 = pop1[gg]
            temp2 = neu_weights_update[vvv]
            a = temp1[:main_feat]
            b = temp2[:main_feat]
            dist = np.linalg.norm(a - b)
            # dist = np.linalg.norm(pop1[gg]-neu_weights_update[vvv])
            distance_test.insert(vvv, dist)
        # print distance_test
        minimum_test = distance_test.index(min(distance_test))  ##Index of winning neuron for test
        distance_test = []  ##Empty the list
        L.insert(gg, minimum_test)  ##Build Mapping of the winning neuron
    # print L
    # print ("\n")
    return L, neu_weights_k


def no_of_neurons(chromosome):
    # pop1 = np.asarray(population)  ##List to Numpy conversion for population
    C = int(math.floor(math.sqrt(chromosome)))  ##C equals floor value of square root(population)
    D = int(C * C)  # no. of neurons
    return C, D


def mapping(S, chromosome, population, tau, t, T, C, D, neu_weights, neu_weights_k, S_K, K, features):
    """
    :param S: Training data for SOM
    :param population:
    :param chromosome:
    :param tau:
    :param t:
    :param T:
    :return:
    """
    pop1 = np.asarray(population)  ##List to Numpy conversion for population
    # C = int(math.floor(math.sqrt(chromosome)))  ##C equals floor value of square root(population)
    # D = int(C * C) #no. of neurons

    # neu_weights = neuron_weights_init(pop1, D)  ##Initial neuron weights

    neu_weights_update, neu_weights_k = neuron_weights_update(S, chromosome, tau, D, C, t, T, neu_weights,
                                                              neu_weights_k, S_K, features)  ##Updated neuron weights

    L, neu_weights_k = build_mapping(chromosome, D, pop1, neu_weights_update, neu_weights_k, K,
                                     features)  ##Mapping of the solution to the neurons

    return neu_weights_update, L, neu_weights_k
