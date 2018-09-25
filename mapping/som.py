import numpy as np
import math

def neuron_weights_init(pop1, D):
    """
        Initializes the weights of neurons from population.
        pop1 - input weight vector(population)
    """
    neu_weights = []    ##Weight of neurons
    for ii in range(0, D):
        neu_weights.insert(ii, pop1[ii])    #Initialize neuron weights from populations

    neu_weights = np.asarray(neu_weights)
    #print neu_weights
    #print("\n\n\n\n\n")
    return neu_weights


def position(minimum, C):
    """
    Calculates the position vector of the neuron
    :param minimum: Index of the winning neuron
    :return: Returns the position vector as x1, x2
    """
    x1 = int(minimum / C) ##x-coordinate of the neuron
    x2 = int(minimum % C) ##y-coordinate for the neuron
    return x1, x2


def eucledian_dist(x1,x2,k1,k2):
    """
    Calculates the eucledian distance between two neurons
    :param x1: x1 co-ordinate of the neuron 1
    :param x2: x2 co-ordinate of the neuron 1
    :param k1: k1 co-ordinate of the neuron 2
    :param k2: k2 co-ordinate of the neuron 2
    :return: returns the eucledian distance
    """
    return math.sqrt(((x1 - k1) ** 2) + (x2 - k2) ** 2)  # Distance between neuron(x1,x2) and (k1,k2)


def neuron_weights_update(S,chromosome, pop1, tau_init, D, C, t, T, neu_weights):
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
    sig_init = C/math.sqrt(2)    ##Sigma initialization
    distance_train = []


    for iii in range(0, len(S)):
        xx = 1 - float(((t-1)*chromosome)+iii)/float(chromosome*T)
        sig = sig_init * xx #Update Sigma
        tau = tau_init * xx  #Update tau
        for jjj in range(0, D):
            dist = np.linalg.norm(S[iii]-neu_weights[jjj])
            distance_train.insert(jjj, dist)    ##List of distance between the current solution and all the neuron
        minimum_train = distance_train.index(min(distance_train))   #Index of the winning neuron for training
        #print distance_train
        #print minimum_train
        distance_train = []     ##Empty the list
        x1, x2 = position(minimum_train, C) ##x,y-coordinate of the winning neuron for train
        for kkk in range(0, D):
            k1, k2 = position(kkk, C)   #x,y-cordinate for the kkkth neighboring neuron
            M = eucledian_dist(x1, x2, k1, k2)  #Distance of winning neuron and kkkth neuron
            if(M < sig):
                #print neu_weights[kkk]
                neu_weights[kkk] += tau * math.exp(-M) * (S[iii] - neu_weights[kkk])   ##Update weights of the neighboring neuron
                #print neu_weights[kkk]
        break
    #print neu_weights
    #print ("\n")
    return neu_weights


def build_mapping(chromosome, D, pop1, neu_weights_update):
    """
    :param neu_weights_update: Updated weight of the neurons
    :return: Returns the Mapping of the population with neurons
    """
    distance_test = []
    L = []  ##Mapping Variable
    for v in range(0, chromosome):
        for vvv in range(0, D):
            dist = np.linalg.norm(pop1[v]-neu_weights_update[vvv])
            distance_test.insert(vvv, dist)
        #print distance_test
        minimum_test = distance_test.index(min(distance_test))  ##Index of winning neuron for test
        #print minimum_test
        distance_test = []      ##Empty the list
        L.insert(v, minimum_test)   ##Build Mapping of the winning neuron
    #print L
    #print ("\n")
    return L

def no_of_neurons(chromosome ):
    #pop1 = np.asarray(population)  ##List to Numpy conversion for population
    C = int(math.floor(math.sqrt(chromosome)))  ##C equals floor value of square root(population)
    D = int(C * C)  # no. of neurons
    return C,D

def mapping(S,population, chromosome, tau, t, T,C,D,neu_weights):
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

    #neu_weights = neuron_weights_init(pop1, D)  ##Initial neuron weights

    neu_weights_update = neuron_weights_update(S,chromosome, pop1, tau, D, C, t, T, neu_weights)  ##Updated neuron weights

    L = build_mapping(chromosome, D, pop1, neu_weights_update)  ##Mapping of the solution to the neurons

    return neu_weights_update, L
