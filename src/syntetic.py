import random 
from scipy.stats import logistic 
import numpy as np 

''' Functions to test our ranking algorithm against a syntethic ground truth'''

def normalize_scores (pi_values):
    norm = 0.0
    val = 0.0
    for n in pi_values:
        norm = norm + np.log(pi_values[n])
        val = val + 1.0

    norm = np.exp(norm/val)

    for n in pi_values:
        pi_values[n] = pi_values[n] / norm


def generate_model_instance (N, M, K1, K2):

    ##random scores from logistic distribution
    pi_values = {}
    for n in range(0, N):
        pi_values[n] = float(np.exp(logistic.rvs(size=1)))


    normalize_scores (pi_values)


    list_of_nodes = list(range(0, N))
    ##
    data = []
    for m in range(0, M):
        K = random.randint(K1, K2)
        tmp = random.sample(list_of_nodes, K)
        ##establish order
        order = establish_order(tmp, pi_values)
        data.append(order)


    return pi_values, data


def establish_order (tmp, pi_values):

    order = []

    while len(tmp) > 0:
        norm = 0.0
        for i in range(0, len(tmp)):
            norm = norm + pi_values[tmp[i]]
        r = random.random() * norm
        norm = 0.0
        s = 0
        for i in range(0, len(tmp)):
            norm =  norm + pi_values[tmp[i]]
            if r > norm:
                s = i + 1

        order.append(tmp[s])
        tmp1 = []
        for i in range(0,len(tmp)):
            if i !=s:
                tmp1.append(tmp[i])
        tmp = tmp1.copy()

    return order


def binarize_data (data):

    bin_data = []

    for i in range(0, len(data)):

        K = len(data[i])
        for r in range(0, K-1):
            for s in range (r+1, K):
                bin_data.append([data[i][r],data[i][s]])


    return bin_data

def create_hypergraph_from_data (data):

    bond_matrix = {}


    for i in range(0, len(data)):

        K = len(data[i])
        for r in range(0, len(data[i])):

            s = data[i][r]

            if s not in bond_matrix:
                bond_matrix[s] = {}
            if K not in bond_matrix[s]:
                bond_matrix[s][K] = {}
            if r not in bond_matrix[s][K]:
                bond_matrix[s][K][r] = []
            bond_matrix[s][K][r].append(data[i])


    return bond_matrix