import random
import numpy as np
import math


def generate_trial(population, q, x1, x2):
    """
    Generates the trial solution as y' = x(q) + F * (x(1) - x(2))
    :param population: Input weight vector(population)
    :param q: Current solution number
    :param x1: Parent 1 selected from the mating pool generated
    :param x2: Parent 2(different than parent 1) selected from the mating pool generated
    :return: Returns the y' value calculated as y' = x(q) + F * (x(1) - x(2))
    """
    F = 0.5  ## DE control parameter, used from experimental setting parameters
    return population[q] + (F * (x1 - x2))  ## Trial solution generation


def repair1(a, b, y1, feat):
    """
    A repair mechanism is adopted to make the solution feasible
    :param a: Lower Boundary
    :param b: Upper Boundary
    :param y1: y1 or y' is the trial solution generated
    :param feat: The number of features that are relevant. It is calculated as |K * data_dimension|
    :return: Returns the y'' value after repairing the solution
    """
    y2 = np.empty(feat)  ## Initialise an empty numpy array y'' of size feat
    # <!--------Repair the solution-------!>
    for i in range(0, feat):
        if y1[i] < a[i]:
            y2[i] = a[i]
        elif y1[i] > b[i]:
            y2[i] = b[i]
        else:
            y2[i] = y1[i]
    return y2


def mutate(a, b, y2, feat):
    """
    A polynomial mutation operator is utilized to enhance the diversity of solutions
    :param a: Lower Boundary
    :param b: Upper Boundary
    :param y2: y'' generated from repairing the solution
    :param feat: The number of features that are relevant. It is calculated as |K * data_dimension|
    :return: Returns the y''' value after mutating the solution
    """
    y3 = np.empty(feat)  ## Initialise a numpy array y''' of size = feat
    pm = (1 / float(feat))  ## Pm defined as 1/n in the experimental setting parameter
    nm = 20  ## Take Nm as 20 as defined in the experimetal setting parameter
    # <!------Mutate the solution-----!>

    for j in range(0, feat):
        r = np.random.uniform(0, 1)  ## Generate a random number r between 0 and 1
        # print r, pm
        if b[j] == a[j]:
            b[j] = 1
            a[j] = 0
        if (r < 0.5):
            delta = math.pow((2 * r) + ((1 - (2 * r)) * math.pow(((b[j] - y2[j]) / (float(b[j] - a[j]))), (nm + 1))),
                             (1 / float(nm + 1))) - 1
            # print r, delta
        else:
            delta = 1 - math.pow(
                (2 - (2 * r)) + (((2 * r) - 1) * math.pow(((y2[j] - a[j]) / (float(b[j] - a[j]))), (nm + 1))),
                (1 / float(nm + 1)))
            # print r, delta

        if (r < pm):
            y3[j] = y2[j] + (delta * (b[j] - a[j]))
        else:
            y3[j] = y2[j]
    return y3


def repair2(a, b, y3, y2, feat):
    """
    A repair operation is again performed on mutated solution
    :param y3: y''' generated from mutating the solution
    :param y2: y'' generated from repairing the solution
    :return: Returns the final solution after repair, mutate and again repair
    """
    y4 = np.empty(feat)  ## Initialise a numpy array y with size = feat
    for k in range(0, feat):
        r = np.random.uniform(0, 1)
        if (y3[k] < a[k]):
            y4[k] = y2[k] - (r * (y2[k] - a[k]))
        elif (y3[k] > b[k]):
            y4[k] = y2[k] + (r * (b[k] - y2[k]))
        else:
            y4[k] = y3[k]
    return y4


def mutate2(y4, data, cluster, q, max_cluster):
    y5 = np.empty(cluster[q] * len(data[0]))  ## Initialise a numpy array y with size = feat
    mut_prob = np.random.uniform(0, 1)  ## Generate a random number mut_prob between 0 and 1
    x = np.random.randint(len(data))
    b = data[x]
    b = np.asarray(b)
    features = len(data[0])
    if mut_prob < 0.6:
        y5 = y4
    elif mut_prob >= 0.6 and mut_prob < 0.8:
        if cluster[q] == max_cluster:
            pass
        else:
            y5 = np.append(y4, b)
    elif mut_prob >= 0.8:
        if cluster[q] == 2:
            pass
        else:
            y5 = y4[:-features].copy()
    return y5


def reproduction(population, q, m_pool, cluster, features, data, max_cluster):
    """
    It produces a new solution y from its current solution x(q) and its mating pool B
    :param population: Input weight vector(population)
    :param q: Current solution number
    :param m_pool: The mating pool that was generated for the current solution x(q)
    :param cluster: A list of all the clusters associated with the population
    :param data: The input data
    :return: Returns the newly generated solution y from the current solution x(q)
    """
    # --------------------Step 1 and Step 2 of reproduction----------------------#
    m_pool = np.asarray(m_pool)  ## List to array conversion of mating pool
    x1, x2 = random.sample(m_pool, 2)  ## Selects two distinct frm mating pool
    y1 = generate_trial(population, q, x1, x2)  ## Call generate_trial for generation of y'
    y1 = np.asarray(y1)  ##  List to array conversion of y'
    feat = cluster[q] * features  ## The number of features that are relevant. It is calculated as |K * data_dimension|
    # print population[q], cluster[q]*len(data[0])
    # print y1
    population_arr = np.asarray(population)
    b = population_arr.max(0)  ## Calculates the upper boundary for all x(i), i = 1, 2, 3, ... n
    b = np.asarray(b)
    a = population_arr.min(0)  ## Calculates the lower boundary for all x(i), i = 1, 2, 3, ... n
    a = np.asarray(a)
    # a=np.zeros(len(population[0]))
    # b=np.ones(len(population[0]))
    # print y1, feat
    # print a[:feat]
    # print b[:feat]
    # ------------------------Repair the solution(Step 3)-------------------------------#

    y2 = repair1(a, b, y1, feat)  ##  Call repair1 for repairing the solution to get y''

    # ------------------------Mutate the solution(Step 4)---------------------------------#
    y3 = mutate(a, b, y2, feat)  ##  Call mutate for mutating the solution to get y'''

    # --------------------Repair the solution again(Step 5)----------------------------#
    y4 = repair2(a, b, y3, y2, feat)  ## Call repair2 for again repairing the solution and getting y=(y1, y2, ... yn)

    # --------------------Mutate the solution again(Step 6)----------------------------#
    y5 = mutate2(y4, data, cluster, q, max_cluster)
    return y5
