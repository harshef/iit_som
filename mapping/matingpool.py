import math
from mapping.matrix import distance_matrix
from mapping.som import no_of_neurons

def mating_pool(q, L, K, v_pos, chromosome, population):
    """
    Creates the mating pool for the solution q
    Note :-
    1). If a neuron is not the winning neuron of any solution, then it will not take part in mating pool construction.
    2). During creation of mating pool for a solution 'q', it excludes itself from its mating pool
    3). During creation of Mating pool for a solution q, the corresponding solution of the Neighboring neurons of winning
        neuron(for q) will take part in mating pool.
    :param q: Current solution number
    :param L: Mapping of the solutions with the winning neuron index
    :param v_pos: A NNS v_pos for x(q) is determined based on idx(k)
    :param chromosome: Number of solution
    :param population: input weight vector(population)
    :return: Returns a mating pool for the current solution x(q) which consists of the corresponding solution of the
             neighboring neurons of the winning neurons
    """
    C,D=no_of_neurons(chromosome)
    # C = int(math.floor(math.sqrt(chromosome)))  ##C equals floor value of square root(population)
    # D = int(C * C)  ##The number of neurons for mapping
    Matrix = distance_matrix(C,D) ##Call distance_matrix for getting a D*D distance matrix between the neurons
    count = 0   ##Counter for iterating the list B(Mating Pool)
    B = []  ##Initialize the list for the mating pool
    B_K = []
    h = L[q]    ##Store the winning neuron mapped by x(q) in h
    I = [i[0] for i in sorted(enumerate(Matrix[h]), key=lambda x:x[1])]   ##Sort hth row of the distance matrix(matrix) and store the indices in I
    H = I[1:v_pos+1]    ##Find out the v_pos neighboring neurons: H = {I(2), ... I(v_pos + 1)}
    for i in range(0, v_pos):
        for j in range(0, chromosome):
            if(H[i]==L[j]): ##Step- 6 i.e if L(k1|k1,k2) = H(i), B = B U {x(k1|k1,k2)}
                B.insert(count, population[j])
                B_K.insert(count, K[j])
                count += 1    ##Increase the iterator
    return B, B_K