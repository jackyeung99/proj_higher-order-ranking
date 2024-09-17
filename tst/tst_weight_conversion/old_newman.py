
import numpy as np


# ===== Old Functions to test against ======
def binarize_data_old(data):

    bin_data = []

    for i in range(0, len(data)):

        K = len(data[i])
        for r in range(0, K-1):
            for s in range (r+1, K):
                bin_data.append([data[i][r],data[i][s]])


    return bin_data



def binarize_data_leadership_old(data):
    bin_data = []
    
    for arr in data:
        arr = np.array(arr)
        pairs = np.column_stack((np.repeat(arr[0], len(arr) - 1), arr[1:]))
        bin_data.extend(pairs.tolist())
        
    return bin_data

def normalize_scores_old(pi_values):
    norm = 0.0
    val = 0.0
    for n in pi_values:
        norm = norm + np.log(pi_values[n])
        val = val + 1.0

    norm = np.exp(norm/val)

    for n in pi_values:
        pi_values[n] = pi_values[n] / norm


def synch_solve_equations_old(bond_matrix, max_iter, pi_values, method, sens=1e-6):

    x, y, z = [], [], []
    scores = {}

    for n in pi_values:
        # scores[n] = float(np.exp(logistic.rvs(size=1)[0]))
        scores[n] = 1.0
   
    normalize_scores_old(scores)
    
    list_of_nodes = list(scores.keys())
    
    
    err = 1.0
    rms = N = 0.0
    for n in scores:
        if n != 'f_p':
            N += 1.0
            rms += (scores[n]-pi_values[n])*(scores[n]-pi_values[n])
    rms = np.sqrt(rms/N)

    x.append(0)
    y.append(rms)
    z.append(err)
    
    iteration = 0
    while iteration < max_iter and err > sens:
        
        err = 0.0
        tmp_scores = {}


        for s in scores:
            tmp_scores[s] = method(s, scores, bond_matrix)

                 
        normalize_scores_old(tmp_scores)

        for s in tmp_scores:
            if abs(tmp_scores[s]-scores[s]) > err:
                err = abs(tmp_scores[s]-scores[s])
            scores[s] = tmp_scores[s]
                
        # print(err)
        iteration += 1
        
        rms = N = 0.0
        for n in scores:
            N += 1.0
            rms += (scores[n]-pi_values[n])*(scores[n]-pi_values[n])
        rms = np.sqrt(rms/N)

        x.append(iteration)
        y.append(rms)
        z.append(err)
   
    return scores, [x, y, z]

def create_hypergraph_from_data_old (data):

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

def iterate_equation_newman_old(s, scores, bond_matrix):
    ##prior
    a = b = 1.0 / (scores[s]+1.0)
    tolerance = 1e-10
    if s in bond_matrix:

        for K in bond_matrix[s]:

            for r in bond_matrix[s][K]:

                if r < K-1:

                    for t in range(0, len(bond_matrix[s][K][r])):
                        tmp1 = tmp2 =  0.0
                        for q in range(r, K):
                            if q > r:
                                tmp1 += scores[bond_matrix[s][K][r][t][q]]
                            tmp2 += scores[bond_matrix[s][K][r][t][q]]

                        if tmp2 != 0:
                            a += tmp1/tmp2


                for t in range(0, len(bond_matrix[s][K][r])):
                    for v in range(0, r):
                        tmp = 0.0
                        for q in range(v, K):
                            tmp += scores[bond_matrix[s][K][r][t][q]]
                     
                        if tmp != 0:
                            b += 1.0 / tmp
                      

  #             for t in range(0, len(bond_matrix[s][K][r])):
  #                 tmp = 0.0
  #                 for q in range(0, K):
  #                     tmp += scores[bond_matrix[s][K][r][t][q]]
  #                 b += 1.0 / tmp
  #                 for q in range(0, r-1):
  #                     tmp = tmp - scores[bond_matrix[s][K][r][t][q]]
  #                     b += 1.0 / tmp

    return a/b



def iterate_equation_newman_leadership_old(s, scores, bond_matrix):

    ##prior
    a = b = 1.0 / (scores[s]+1.0)
    tolerance = 1e-10
    if s in bond_matrix:

        for K in bond_matrix[s]:
            
    #         print (bond_matrix[s][K])
            
            for r in bond_matrix[s][K]:

                if r == 0:
                    
    #                 print (bond_matrix[s][K][r])
                
                    for t in range(0, len(bond_matrix[s][K][r])):
                        tmp1 = tmp2 =  0.0
                        for q in range(0, K):
                            if q>0:
                                tmp1 += scores[bond_matrix[s][K][r][t][q]]
                            tmp2 += scores[bond_matrix[s][K][r][t][q]]

                        if tmp2 != 0:
                            a += tmp1/tmp2
                    
                else:
                    for t in range(0, len(bond_matrix[s][K][r])):
                        tmp = 0.0
                        for q in range(0, K): 
                            tmp += scores[bond_matrix[s][K][r][t][q]]

                        if tmp != 0:
                            b += 1.0 / tmp

    return a/b

def compute_predicted_ratings_BT_old(training_set, pi_values):
    bin_data = binarize_data_old(training_set)
    bin_bond_matrix = create_hypergraph_from_data_old(bin_data)

    predicted_std_scores, _ = synch_solve_equations_old(bin_bond_matrix, 10000, pi_values, iterate_equation_newman_old, sens=1e-6)

    return predicted_std_scores

def compute_predicted_ratings_BT_leadership_old(training_set, pi_values): 
    bin_data = binarize_data_leadership_old(training_set)
    bin_bond_matrix = create_hypergraph_from_data_old(bin_data)

    predicted_std_scores, _ = synch_solve_equations_old(bin_bond_matrix, 10000, pi_values, iterate_equation_newman_old, sens=1e-6)

    return predicted_std_scores

def compute_predicted_ratings_HO_BT_old(training_set, pi_values): 
    
    bond_matrix = create_hypergraph_from_data_old(training_set)
    predicted_ho_scores, _ = synch_solve_equations_old(bond_matrix, 10000, pi_values, iterate_equation_newman_old, sens=1e-6)

    return predicted_ho_scores


def compute_predicted_ratings_HOL_BT_old(training_set, pi_values):
    bond_matrix = create_hypergraph_from_data_old(training_set)
    predicted_hol_scores, _ = synch_solve_equations_old(bond_matrix, 10000, pi_values, iterate_equation_newman_leadership_old, sens=1e-6)

    return predicted_hol_scores
