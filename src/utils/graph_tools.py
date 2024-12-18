import os 
import sys
import random 
from scipy.stats import logistic 
import numpy as np 


# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(repo_root)

''' Functions to represent ordered games into hypergraph structure '''

def create_hypergraph_from_data(data):

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

def binarize_data(data):

    bin_data = []

    for i in range(0, len(data)):

        K = len(data[i])
        for r in range(0, K-1):
            for s in range (r+1, K):
                bin_data.append([data[i][r],data[i][s]])


    return bin_data



def binarize_data_leadership(data):
    bin_data = []
    
    for arr in data:
        arr = np.array(arr)
        pairs = np.column_stack((np.repeat(arr[0], len(arr) - 1), arr[1:]))
        bin_data.extend(pairs.tolist())
        
    return bin_data



def normalize_scores(pi_values):
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
    for n in range(1, N+1):
        pi_values[n] = float(np.exp(logistic.rvs(size=1)[0]))


    normalize_scores (pi_values)


    list_of_nodes = list(range(1, N+1))
    ##
    data = []
    for m in range(0, M):
        K = random.randint(K1, K2)
        tmp = random.sample(list_of_nodes, K)
        ##establish order
        order = establish_order(tmp, pi_values)
        data.append(order)


    return data, pi_values

def generate_leadership_model_instance (N, M, K1, K2):
    
    ##random scores from logistic distribution
    pi_values = {}
    for n in range(1, N+1):
        pi_values[n] = float(np.exp(logistic.rvs(size=1)[0]))

    
    normalize_scores (pi_values)
    
    
    list_of_nodes = list(range(1, N+1))
    ##
    data = []
    for m in range(0, M):
        K = random.randint(K1, K2)
        tmp = random.sample(list_of_nodes, K)
        ##establish order
        order = establish_order (tmp, pi_values)
        #print (order)
        f = order[0]
        order = order[1:]
        random.shuffle(order)
        order.insert(0,f)
        data.append(order)
        #print (order,'\n')
        
        
    return data, pi_values




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



def convert_dict_to_games(weighted_games):

    list_of_games = []
    for ordering, count in weighted_games.items():
        list_of_games.extend([list(ordering)] * count)
        
    return list_of_games
