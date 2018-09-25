import math
import numpy as np


def distance_matrix(C,D):
    """
    Calculates the distance matrix between each neuron and every other neuro
    :param C:
    :param D: The number of neurons
    :return: Returns the distance matrix for the D neurons
    """
    Matrix = [[0 for x in range(D)] for y in range(D)]  ##Initialize a two dimensional distance matrix of size D*D
    M = []  ##Initialize a list for storing co-ordinates
    a = C   ##For distributing the neurons index to co-ordinates
    for i in range(0, D):
        x1 = i/a    ##x1 co-ordinate for each neuron
        x2 = i%a    ##x2 co-ordinate for each neuron
        M.insert(i, (x1, x2))     ##Insert the co-ordinates of each neurons in a list
    M = np.asarray(M)   ##List to numpy conversion
    for j in range(0, D):
        for k in range(0, D):
            Matrix[j][k] = np.linalg.norm(M[j] - M[k])  ##Find the distance between each neuron and every other neuron and create a distance matrix
    return Matrix