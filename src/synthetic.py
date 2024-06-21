import random 
from scipy.stats import logistic 
import numpy as np 

''' Functions to test our ranking algorithm against a syntethic ground truth'''




def generate_model_instance (N, M, K1, K2):

    ##random scores from logistic distribution
    pi_values = {}
    for n in range(0, N):
        pi_values[n] = float(np.exp(logistic.rvs(size=1)))


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


    return pi_values, data

def generate_leadership_model_instance (N, M, K1, K2):
        
        ##random scores from logistic distribution
        pi_values = {}
        for n in range(0, N):
            pi_values[n] = float(np.exp(logistic.rvs(size=1)))

        
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
            
            
        return pi_values, data 



# ============= Creating Fraphs from Sampled Games =============
def create_hypergraph_from_data (data):

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


def binarize_data (data):

    bin_data = []

    for i in range(0, len(data)):

        K = len(data[i])
        for r in range(0, K-1):
            for s in range (r+1, K):
                bin_data.append([data[i][r],data[i][s]])


    return bin_data

def binarize_data_leadership (data):
    
    bin_data = []
    
    for i in range(0, len(data)):
        
        K = len(data[i])
        for s in range (1, K):
            bin_data.append([data[i][0],data[i][s]])
        
        
    return bin_data