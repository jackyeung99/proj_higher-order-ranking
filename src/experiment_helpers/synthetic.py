import os 
import sys
import random 
from scipy.stats import logistic 
import numpy as np 


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_root)

from src.utils import *
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

