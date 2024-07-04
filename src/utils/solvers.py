from scipy.stats import logistic
import numpy as np 
import random 
import os 
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_root)



def synch_solve_equations (bond_matrix, max_iter, pi_values, method, sens=1e-10):

    x, y, z = [], [], []
    scores = {}
#     for n in bond_matrix:
    for n in pi_values:
        # scores[n] = random.random()
        scores[n] = float(np.exp(logistic.rvs(size=1)))
        # scores[n] = 1.0
        # scores[n] = pi_values[n]
        # if n not in bond_matrix:
        #     bond_matrix[n] = {}
    normalize_scores (scores)
    
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
            
                            
        normalize_scores (tmp_scores)
        
        for s in tmp_scores:
            if abs(tmp_scores[s]-scores[s]) > err:
                err = abs(tmp_scores[s]-scores[s])
            scores[s] = tmp_scores[s]
                
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

def solve_equations (bond_matrix, max_iter, pi_values, method, sens=1e-6):

    ''' 
    Offline/asynchronous iterative method:
    update rankings of players in batchs
    '''


    x, y, z = [], [], []


    scores = {}
    for n in bond_matrix:
        scores[n] = random.random()
        normalize_scores (scores)


    list_of_nodes = list(scores.keys())


    err = 1.0
    rms = N = 0.0
    for n in scores:
        N += 1.0
        rms += (scores[n]-pi_values[n])*(scores[n]-pi_values[n])
    rms = np.sqrt(rms/N)

    x.append(0)
    y.append(rms)
    z.append(err)



    iteration = 0
    while iteration < max_iter and err > sens:

        err = 0.0

        for n in range(0, len(scores)):

            s = random.choice(list_of_nodes)

            old = scores[s]
            scores[n] = method(s, scores, bond_matrix)
            normalize_scores (scores)

            if abs(old-scores[s]) > err:
                err = abs(old-scores[s])




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

def normalize_scores (pi_values):
    norm = 0.0
    val = 0.0
    for n in pi_values:
        norm = norm + np.log(pi_values[n])
        val = val + 1.0

    norm = np.exp(norm/val)

    for n in pi_values:
        pi_values[n] = pi_values[n] / norm

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

