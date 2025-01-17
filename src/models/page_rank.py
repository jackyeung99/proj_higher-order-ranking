import os 
import sys
import networkx as nx 

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.graph_tools import binarize_data, binarize_data_leadership, normalize_scores


def compute_predicted_ratings_page_rank(games, pi_values):

    
    edge_list = binarize_data(games)

    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    
    # Initialize PageRank values with the provided pi_values
    personalization = {node: 1.0 for node in G.nodes}
    
    page_rank = nx.pagerank(G, personalization=personalization)
    pred_rankings = {}
    for node in pi_values:
        pred_rankings[node] = page_rank.get(node, 1.0)
        
        
    return pred_rankings


def compute_predicted_ratings_page_rank_leadership(games, pi_values):

    
    edge_list = binarize_data_leadership(games)

    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    
    # Initialize PageRank values with the provided pi_values
    personalization = {node: 1.0 for node in G.nodes}
    
    page_rank = nx.pagerank(G, personalization=personalization)
    pred_rankings = {}
    for node in pi_values:
        pred_rankings[node] = page_rank.get(node, 1.0)
        
        
    return pi_values




