import os 
import sys
import random 
from scipy.stats import logistic 
import numpy as np 


# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(repo_root)

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
        pi_values[n] = float(np.exp(logistic.rvs(size=1)[0]))


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


    return data, pi_values

def generate_leadership_model_instance (N, M, K1, K2):
    
    ##random scores from logistic distribution
    pi_values = {}
    for n in range(0, N):
        pi_values[n] = float(np.exp(logistic.rvs(size=1)[0]))

    
    normalize_scores (pi_values)
    
    
    list_of_nodes = list(range(0, N))
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

def generate_model_instance_weighted(N, M, K1, K2):
    # Random scores from logistic distribution
    pi_values = {n: float(np.exp(logistic.rvs(size=1)[0])) for n in range(N)}
    normalize_scores(pi_values)

    list_of_nodes = list(range(N))
    data = {}

    for m in range(M):
        K = random.randint(K1, K2)
        tmp = random.sample(list_of_nodes, K)
        order = tuple(establish_order(tmp, pi_values))
        if order not in data:
            data[order] = 0

        data[order] += 1 

    return data, pi_values

def generate_leadership_model_instance_weighted(N, M, K1, K2):
    # Random scores from logistic distribution
    pi_values = {n: float(np.exp(logistic.rvs(size=1)[0])) for n in range(N)}
    normalize_scores(pi_values)

    list_of_nodes = list(range(N))
    data = {}

    for m in range(M):
        K = random.randint(K1, K2)
        tmp = random.sample(list_of_nodes, K)
        order = establish_order(tmp, pi_values)
        
        f = order[0]
        order = order[1:]
        random.shuffle(order)
        order.insert(0, f)
        order = tuple(order)

        if order not in data:
            data[order] = 0
            
        data[order] += 1 
       
    return data, pi_values
 
def create_hypergraph_from_data_weight(data):
    bond_matrix = {}
    for game, weight in data.items():
        K = len(game)
        for position, player in enumerate(game):
            if player not in bond_matrix:
                bond_matrix[player] = []
            bond_matrix[player].append((K, position, game, weight))

    return bond_matrix


# def create_hypergraph_from_data_weight (data):

#     bond_matrix = {}


#     for order, weight in data.items():

#         K = len(order)
#         for player in order:
            
#             position = order.index(player)

#             if player not in bond_matrix:
#                 bond_matrix[player] = {}
#             if K not in bond_matrix[player]:
#                 bond_matrix[player][K] = {}
#             if position not in bond_matrix[player][K]:
#                 bond_matrix[player][K][position] = {}

#             bond_matrix[player][K][position][order] = weight


#     return bond_matrix

def binarize_data_weighted(data):
    bin_data = {}
  
    for game, weight in data.items():
        arr = np.array(game)
        idx = np.triu_indices(len(arr), k=1)
        pairs = np.array([arr[idx[0]], arr[idx[1]]]).T
        for pair in pairs:
            pair_tuple = tuple(pair)
            if pair_tuple in bin_data:
                bin_data[pair_tuple] += weight
            else:
                bin_data[pair_tuple] = weight

    return bin_data


def binarize_data_weighted_leadership (data):
    
    bin_data = {}
    
    for arr, weight in data.items():
        arr = np.array(arr)
        pairs = np.column_stack((np.repeat(arr[0], len(arr) - 1), arr[1:]))
        for pair in pairs:
            pair_tuple = tuple(pair)
            if pair_tuple in bin_data:
                bin_data[pair_tuple] += weight
            else:
                bin_data[pair_tuple] = weight

    return bin_data


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

def convert_games_to_dict(games):
    # Count occurrences of each unique ordering
    unique_orderings = {}
    for game in games:
        ordering = tuple(game)  
        if ordering in unique_orderings:
            unique_orderings[ordering] += 1
        else: 
            unique_orderings[ordering] = 1
    
    return dict(sorted(unique_orderings.items(), key = lambda x:x[1],  reverse=True))


