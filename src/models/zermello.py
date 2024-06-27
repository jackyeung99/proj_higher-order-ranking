import random 
from ..utils.graph_tools import * 



def iterate_equation_zermelo (s, scores, bond_matrix):


    #a = b = 0.0

    ##prior
    a = 1.0
    b = 2.0/(scores[s]+1.0)

    for K in bond_matrix[s]:



        for r in bond_matrix[s][K]:


            if r < K-1:
                a += len(bond_matrix[s][K][r])

                for t in range(0, len(bond_matrix[s][K][r])):
                    tmp = 0.0
                    for q in range(r, K):
                        tmp += scores[bond_matrix[s][K][r][t][q]]
                    b += 1.0 / tmp


            for t in range(0, len(bond_matrix[s][K][r])):
                for v in range(0, r):
                    tmp = 0.0
                    for q in range(v, K):
                        tmp += scores[bond_matrix[s][K][r][t][q]]
                    b += 1.0 / tmp
#                     print ('> ', tmp)


#             for t in range(0, len(bond_matrix[s][K][r])):
#                 tmp = 0.0
#                 for q in range(0, K):
#                     tmp += scores[bond_matrix[s][K][r][t][q]]
#                 b += 1.0 / tmp
#                 print ('>> ', tmp)
#                 for q in range(0, r-1):
#                     tmp = tmp - scores[bond_matrix[s][K][r][t][q]]
#                     print ('>> ', tmp)
#                     b += 1.0 / tmp


    return a/b



def iterate_equation_zermelo_new (s, scores, bond_matrix):


    #a = b = 0.0

    ##prior
    a = 1.0
    b = 2.0/(scores[s]+1.0)

    for K in bond_matrix[s]:



        for r in bond_matrix[s][K]:



            a += len(bond_matrix[s][K][r])



            for t in range(0, len(bond_matrix[s][K][r])):
                for v in range(0, r+1):
                    tmp = 0.0
                    for q in range(v, K):
                        tmp += scores[bond_matrix[s][K][r][t][q]]
                    b += 1.0 / tmp


    return a/b



