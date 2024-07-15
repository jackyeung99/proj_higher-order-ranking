import os 
import sys
import networkx as nx 

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.graph_tools import *

def compute_predicted_ratings_page_rank(games, true_pi_values):
    bin_data = binarize_data_weighted(games)

    weighted_edges = [(i,j,weight) for (i,j), weight in bin_data.items()]
    
    G = nx.DiGraph()
    G.add_weighted_edges_from(weighted_edges)
    personalization = {node: 1.0 for node in G.nodes}
    
    page_rank = nx.pagerank(G, personalization=personalization)

    for player in true_pi_values:
        if player not in page_rank:
            page_rank[player] = 1.0

    normalize_scores(page_rank)
        
    return page_rank


def compute_predicted_ratings_page_rank_leadership(games, true_pi_values):
    bin_data = binarize_data_weighted_leadership(games)
    weighted_edges = [(i,j,weight) for (i,j), weight in bin_data.items()]
        
    G = nx.DiGraph()
    G.add_weighted_edges_from(weighted_edges)
    personalization = {node: 1.0 for node in G.nodes}
    
    page_rank = nx.pagerank(G, personalization=personalization)

    for player in true_pi_values:
        if player not in page_rank:
            page_rank[player] = 1.0

    normalize_scores(page_rank)
        
    return page_rank

