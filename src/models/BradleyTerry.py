import sys
import os
import numpy as np
from scipy.stats import logistic


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.graph_tools import normalize_scores, binarize_data, binarize_data_leadership, create_hypergraph_from_data

MAX_ITER = 10000
EPS = 1e-6


def rms_error(new_scores, old_scores):
    # calculate the probablity that a player beats the average player for old and new iterated scores
    beating_avg_new = [(pi / (pi + 1)) for pi in new_scores.values()]
    beating_avg_old = [(pi / (pi + 1)) for pi in old_scores.values()]

    # Compute the RMS error between the transformed scores
    return np.sqrt(np.mean([(new - old) ** 2 for new, old in zip(beating_avg_new, beating_avg_old)]))


def std_error(new_scores, old_scores):
    err = 0
    for s in old_scores:
        cur_err = abs(np.log(new_scores[s])-np.log(old_scores[s]))
        err = max(cur_err, err)

    return err


def random_number_from_logistic():
    return 1.0 / np.random.rand() - 1.0



# Modified Solver function  to Keep Track of iterations for Genralized Newman 
def synch_solve_equations(hypergraph, pi_values, method, max_iter=MAX_ITER, sens=EPS, convergence_metric='rms'):
    # logistic_distribution = np.sqrt(np.exp(logistic.rvs(size=len(pi_values))))
    scores = {n: random_number_from_logistic () for n in pi_values}
    normalize_scores(scores)

    err = 1.0

    info = {}
    iteration = 0
    while iteration < max_iter and err > sens:
        tmp_scores = {s: method(s, scores, hypergraph) for s in scores}
                 
        normalize_scores(tmp_scores)

        if convergence_metric == 'rms':
            err = rms_error(tmp_scores, scores)
        else:
            err = std_error(tmp_scores, scores)

        scores = tmp_scores.copy()

        iteration += 1
        info[iteration] = err
 
    return scores, info


def iterate_equation_newman(s, scores, hypergraph):
    ##prior
    a = b = 1.0 / (scores[s]+1.0)
    if s in hypergraph:

        for K in hypergraph[s]:

            for r in hypergraph[s][K]:

                if r < K-1:

                    for t in range(0, len(hypergraph[s][K][r])):
                        tmp1 = tmp2 =  0.0
                        for q in range(r, K):
                            if q > r:
                                tmp1 += scores[hypergraph[s][K][r][t][q]]
                            tmp2 += scores[hypergraph[s][K][r][t][q]]

                        if tmp2 != 0:
                            a += tmp1/tmp2


                for t in range(0, len(hypergraph[s][K][r])):
                    for v in range(0, r):
                        tmp = 0.0
                        for q in range(v, K):
                            tmp += scores[hypergraph[s][K][r][t][q]]
                     
                        if tmp != 0:
                            b += 1.0 / tmp
                      

  #             for t in range(0, len(hypergraph[s][K][r])):
  #                 tmp = 0.0
  #                 for q in range(0, K):
  #                     tmp += scores[hypergraph[s][K][r][t][q]]
  #                 b += 1.0 / tmp
  #                 for q in range(0, r-1):
  #                     tmp = tmp - scores[hypergraph[s][K][r][t][q]]
  #                     b += 1.0 / tmp

    return a/b



def iterate_equation_newman_leadership(s, scores, hypergraph):

    ##prior
    a = b = 1.0 / (scores[s]+1.0)
    if s in hypergraph:

        for K in hypergraph[s]:
            
    #         print (hypergraph[s][K])
            
            for r in hypergraph[s][K]:

                if r == 0:
                    
    #                 print (hypergraph[s][K][r])
                
                    for t in range(0, len(hypergraph[s][K][r])):
                        tmp1 = tmp2 =  0.0
                        for q in range(0, K):
                            if q>0:
                                tmp1 += scores[hypergraph[s][K][r][t][q]]
                            tmp2 += scores[hypergraph[s][K][r][t][q]]

                        if tmp2 != 0:
                            a += tmp1/tmp2
                    
                else:
                    for t in range(0, len(hypergraph[s][K][r])):
                        tmp = 0.0
                        for q in range(0, K): 
                            tmp += scores[hypergraph[s][K][r][t][q]]

                        if tmp != 0:
                            b += 1.0 / tmp

    return a/b

def compute_predicted_ratings_BIN(training_set, pi_values, verbose=False):
    bin_data = binarize_data(training_set)
    hyper_graph = create_hypergraph_from_data(bin_data)

    predicted_scores, info = synch_solve_equations(hyper_graph, pi_values, iterate_equation_newman)

    if verbose:
        return predicted_scores, info
    else:
        return predicted_scores

def compute_predicted_ratings_BINL(training_set, pi_values, verbose=False): 
    bin_data = binarize_data_leadership(training_set)
    hyper_graph = create_hypergraph_from_data(bin_data)

    predicted_scores, info = synch_solve_equations(hyper_graph, pi_values, iterate_equation_newman)

    if verbose:
        return predicted_scores, info
    else:
        return predicted_scores

def compute_predicted_ratings_HO_BT(training_set, pi_values, verbose=False): 
    
    hyper_graph = create_hypergraph_from_data(training_set)
    predicted_scores, info = synch_solve_equations(hyper_graph, pi_values, iterate_equation_newman)

    if verbose:
        return predicted_scores, info
    else:
        return predicted_scores


def compute_predicted_ratings_HOL_BT(training_set, pi_values, verbose=False):
    hyper_graph = create_hypergraph_from_data(training_set)
    predicted_scores, info = synch_solve_equations(hyper_graph, pi_values, iterate_equation_newman_leadership)

    if verbose:
        return predicted_scores, info
    else:
        return predicted_scores