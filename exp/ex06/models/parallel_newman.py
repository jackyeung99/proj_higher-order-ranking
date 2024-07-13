import os
import sys
import numpy as np
import concurrent.futures

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils import *
from datasets.utils.extract_ordered_games import *

# ===== Old Functions to test against ======

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

def normalize_scores(pi_values):
    log_values = np.log(list(pi_values.values()))
    norm = np.exp(np.mean(log_values))
    for n in pi_values:
        pi_values[n] /= norm

def synch_solve_equations_old(bond_matrix, max_iter, pi_values, method, sens=1e-10):
    x, y, z = [], [], []
    scores = {n: 1.0 for n in pi_values}
    normalize_scores(scores)

    list_of_nodes = list(scores.keys())
    rms = np.sqrt(np.mean([(scores[n] - pi_values[n]) ** 2 for n in scores]))
    x.append(0)
    y.append(rms)
    z.append(1.0)
    
    iteration = 0
    err = 1.0
    
    while iteration < max_iter and err > sens:
        err = 0.0
        tmp_scores = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(method, s, scores, bond_matrix): s for s in scores}
            for future in concurrent.futures.as_completed(futures):
                s = futures[future]
                tmp_scores[s] = future.result()

        normalize_scores(tmp_scores)

        err = max(abs(tmp_scores[s] - scores[s]) for s in tmp_scores)
        scores.update(tmp_scores)
        
        rms = np.sqrt(np.mean([(scores[n] - pi_values[n]) ** 2 for n in scores]))
        x.append(iteration + 1)
        y.append(rms)
        z.append(err)
        
        iteration += 1

    return scores, [x, y, z]

def create_hypergraph_from_data(data):
    bond_matrix = {}
    for arr in data:
        K = len(arr)
        for r, s in enumerate(arr):
            bond_matrix.setdefault(s, {}).setdefault(K, {}).setdefault(r, []).append(arr)
    return bond_matrix

def iterate_equation_newman(s, scores, bond_matrix):
    a = b = 1.0 / (scores[s] + 1.0)
    if s in bond_matrix:
        for K in bond_matrix[s]:
            for r in bond_matrix[s][K]:
                if r < K - 1:
                    for t in bond_matrix[s][K][r]:
                        tmp1 = tmp2 = 0.0
                        for q in range(r, K):
                            if q > r:
                                tmp1 += scores[t[q]]
                            tmp2 += scores[t[q]]
                        a += tmp1 / tmp2
                for t in bond_matrix[s][K][r]:
                    for v in range(0, r):
                        tmp = sum(scores[t[q]] for q in range(v, K))
                        b += 1.0 / tmp
    return a / b

def iterate_equation_newman_leadership(s, scores, bond_matrix):
    a = b = 1.0 / (scores[s] + 1.0)
    if s in bond_matrix:
        for K in bond_matrix[s]:
            for r in bond_matrix[s][K]:
                if r == 0:
                    for t in bond_matrix[s][K][r]:
                        tmp1 = tmp2 = 0.0
                        for q in range(K):
                            if q > 0:
                                tmp1 += scores[t[q]]
                            tmp2 += scores[t[q]]
                        a += tmp1 / tmp2
                else:
                    for t in bond_matrix[s][K][r]:
                        tmp = sum(scores[t[q]] for q in range(K))
                        b += 1.0 / tmp
    return a / b

def parrallel_std(training_set, pi_values):
    bin_data = binarize_data(training_set)
    bin_bond_matrix = create_hypergraph_from_data(bin_data)
    predicted_std_scores, _ = synch_solve_equations_old(bin_bond_matrix, 1000, pi_values, iterate_equation_newman, sens=1e-10)

    return predicted_std_scores

def parallel_std_leadership(training_set, pi_values):
    bin_data = binarize_data_leadership(training_set)
    bin_bond_matrix = create_hypergraph_from_data(bin_data)
    predicted_std_scores, _ = synch_solve_equations_old(bin_bond_matrix, 1000, pi_values, iterate_equation_newman, sens=1e-10)
    return predicted_std_scores

def parallel_ho(training_set, pi_values):
    bond_matrix = create_hypergraph_from_data(training_set)
    predicted_ho_scores, _ = synch_solve_equations_old(bond_matrix, 1000, pi_values, iterate_equation_newman, sens=1e-10)

    return predicted_ho_scores

def parallel_hol(training_set, pi_values):
    bond_matrix = create_hypergraph_from_data(training_set)
    predicted_hol_scores, _ = synch_solve_equations_old(bond_matrix, 1000, pi_values, iterate_equation_newman_leadership, sens=1e-10)
    return predicted_hol_scores