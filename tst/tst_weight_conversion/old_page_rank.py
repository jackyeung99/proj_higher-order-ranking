import os 
import sys
import networkx as nx 

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.graph_tools import *

def binarize_data(data):
    bin_data = []
    for arr in data:
        arr = np.array(arr)
        idx = np.triu_indices(len(arr), k=1)
        pairs = np.array([arr[idx[0]], arr[idx[1]]]).T
        bin_data.extend(pairs.tolist())
    return bin_data

def binarize_data_leadership(data):
    bin_data = []
    
    for arr in data:
        arr = np.array(arr)
        pairs = np.column_stack((np.repeat(arr[0], len(arr) - 1), arr[1:]))
        bin_data.extend(pairs.tolist())
        
    return bin_data

def compute_predicted_ratings_page_rank_old(games, pi_values):
    edge_list = binarize_data (games)

    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    
    # Initialize PageRank values with the provided pi_values
    personalization = {node: 1.0 for node in G.nodes}
    
    page_rank = nx.pagerank(G, personalization=personalization)
    for node in pi_values:
        pi_values[node] = page_rank.get(node, 1.0)
        
    
    normalize_scores(pi_values)
        
    return pi_values


def compute_predicted_ratings_page_rank_leadership_old(games, pi_values):
    edge_list = binarize_data_leadership (games)

    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    
    # Initialize PageRank values with the provided pi_values
    personalization = {node: 1.0 for node in G.nodes}
    
    page_rank = nx.pagerank(G, personalization=personalization)
    for node in pi_values:
        pi_values[node] = page_rank.get(node, 1.0)
            
    
    normalize_scores(pi_values)
        
    return pi_values