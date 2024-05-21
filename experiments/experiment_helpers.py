import math
from src.solvers import *
from src.syntetic import * 

def rms(predicted_scores, ground_truth_scores):
    # Calculate the root mean square error
    return np.sqrt(np.mean((np.array(predicted_scores) - np.array(ground_truth_scores)) ** 2))


def generate_test_data(N, M, K1, K2):
    # Generate synthetic data
    pi_values, interactions = generate_model_instance(N, M, K1, K2)

    # Create hypergraph
    bond_matrix = create_hypergraph_from_data(interactions)

    # Standard graph
    bin_data = binarize_data(interactions)
    bin_bond_matrix = create_hypergraph_from_data(bin_data)

    return pi_values, interactions, bond_matrix, bin_bond_matrix


def run_experiments(test_parameters, independent):

    all_results = []

    for param_set in test_parameters:
        N = param_set['N']
        M = param_set['M']
        K1 = param_set['K1']
        K2 = param_set['K2']

        pi_values, test_hyper_graph, test_graph = generate_test_data(N,M,K1,K2)

        # Measure accuracy for hypergraph and standard graph
        hyper_graph_accuracy = rms(test_hyper_graph, pi_values)
        graph_accuracy = rms(test_graph, pi_values)


        all_results.append({
            'graph_accuracy': graph_accuracy,
            'hyper_graph_accuracy': hyper_graph_accuracy
        })


    return all_results


