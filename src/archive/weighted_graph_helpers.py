import os 
import sys
import numpy as np
from scipy.stats import logistic 

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.graph_tools import normalize_scores

def binarize_data_weighted(data):
    bin_data = {}
    
    for game, weight in data.items():
        if len(game) > 2:
            arr = np.array(game)
            idx = np.triu_indices(len(arr), k=1)
            pairs = np.array([arr[idx[0]], arr[idx[1]]]).T
            for pair in pairs:
                pair_tuple = tuple(pair)
                bin_data[pair_tuple] = bin_data.get(pair_tuple, 0) + weight
        else:
            bin_data[game] = bin_data.get(game, 0) + weight

    return bin_data


def binarize_data_weighted_leadership(data):
    bin_data = {}
    
    for game, weight in data.items():
        if len(game) > 2:
            arr = np.array(game)
            pairs = np.column_stack((np.repeat(arr[0], len(arr) - 1), arr[1:]))
            for pair in pairs:
                pair_tuple = tuple(pair)
                bin_data[pair_tuple] = bin_data.get(pair_tuple, 0) + weight
        else:
            bin_data[game] = bin_data.get(game, 0) + weight

    return bin_data

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

def create_hypergraph_from_data_weight(data):
    bond_matrix = {}
    for game, weight in data.items():
        K = len(game)
        for position, player in enumerate(game):
            if player not in bond_matrix:
                bond_matrix[player] = []
            bond_matrix[player].append((K, position, game, weight))

    return bond_matrix


def generate_weighted_model_instance(N, M, K1, K2):
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

def generate_weighted_leadership_model_instance(N, M, K1, K2):
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
 