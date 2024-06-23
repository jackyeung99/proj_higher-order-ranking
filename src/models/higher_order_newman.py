import sys
import os
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.models.newman import *
from src.experiment_helpers.file_handlers import *
from src.experiment_helpers.metrics import * 
from src.experiment_helpers.synthetic import *
from src.utils import *


def iterate_equation_newman_leadership (s, scores, bond_matrix):

    ##prior
    a = b = 1.0 / (scores[s]+1.0)
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

                        a += tmp1/tmp2
                    
                else:
                    for t in range(0, len(bond_matrix[s][K][r])):
                        tmp = 0.0
                        for q in range(0, K): 
                            tmp += scores[bond_matrix[s][K][r][t][q]]
                        b += 1.0 / tmp

    return a/b

def compute_predicted_rankings_ho(training_set, pi_values): 
    
    bond_matrix = create_hypergraph_from_data (training_set)
    predicted_ho_scores, _ = synch_solve_equations(bond_matrix, 1000, pi_values, iterate_equation_newman, sens=1e-6)

    return predicted_ho_scores



def compute_predicted_rankings_hol(training_set, pi_values):
    bond_matrix = create_hypergraph_from_data (training_set)
    predicted_hol_scores, _ = synch_solve_equations (bond_matrix, 1000, pi_values, iterate_equation_newman_leadership, sens=1e-6)

    return predicted_hol_scores


if __name__ == '__main__':
    
    data, pi_values = read_strict_ordered_dataset('datasets/processed_data/00045-00000014.soc')


    train, test = train_test_split(data)
    
    ratings = compute_predicted_rankings_ho(train, pi_values)

    print(compute_likelihood(ratings, test))