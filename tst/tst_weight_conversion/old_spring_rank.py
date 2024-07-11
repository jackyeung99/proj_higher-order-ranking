import numpy as np
from numba import jit
from scipy.sparse import spdiags, csr_matrix
from scipy.optimize import brentq
import scipy.sparse.linalg
import os 
import sys
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.graph_tools import *


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

''' Credit to https://github.com/LarremoreLab/SpringRank for this file'''
def get_scaled_ranks(A, scale=0.75):
    """
    params:
    - A: a (square) np.ndarray
    - scale: float, greater than 0 less than 1; scale is the probability 
        that a node x with rank R_x will beat nodes with rank R_x - 1

    returns:
    - ranks, np.array

    TODO:
    - support passing in other formats (eg a sparse matrix)
    """
    ranks = get_ranks(A)
    inverse_temperature = get_inverse_temperature(A, ranks)
    scaling_factor = 1 / (np.log(scale / (1 - scale)) / (2 * inverse_temperature))
    scaled_ranks = scale_ranks(ranks,scaling_factor)
    return scaled_ranks


def get_ranks(A):
    """
    params:
    - A: a (square) np.ndarray

    returns:
    - ranks, np.array

    TODO:
    - support passing in other formats (eg a sparse matrix)
    """
    return SpringRank(A)


def get_inverse_temperature(A, ranks):
    """given an adjacency matrix and the ranks for that matrix, calculates the
    temperature of those ranks"""
    betahat = brentq(eqs39, 0.01, 20, args=(ranks, A))
    return betahat


def scale_ranks(ranks, scaling_factor):
    return ranks * scaling_factor


def csr_SpringRank(A):
    """
    Main routine to calculate SpringRank by solving linear system
    Default parameters are initialized as in the standard SpringRank model

    Arguments:
        A: Directed network (np.ndarray, scipy.sparse.csr.csr_matrix)

    Output:
        rank: N-dim array, indeces represent the nodes' indices used in ordering the matrix A
    """

    N = A.shape[0]
    k_in = np.array(A.sum(axis=0))
    k_out = np.array(A.sum(axis=1).transpose())

    # form the graph laplacian
    operator = csr_matrix(
        spdiags(k_out + k_in, 0, N, N) - A - A.transpose()
    )

    # form the operator A (from Ax=b notation)
    # note that this is the operator in the paper, but augmented
    # to solve a Lagrange multiplier problem that provides the constraint
    operator.resize((N + 1, N + 1))
    operator[N, 0] = 1
    operator[0, N] = 1

    # form the solution vector b (from Ax=b notation)
    solution_vector = np.append((k_out - k_in), np.array([0])).transpose()

    # perform the computations
    ranks = scipy.sparse.linalg.bicgstab(
        scipy.sparse.csr_matrix(operator),
        solution_vector
    )[0]

    return ranks[:-1]


def SpringRank(A, alpha=0):
    """
    Solve the SpringRank system.
    If alpha = 0, solves a Lagrange multiplier problem.
    Otherwise, performs L2 regularization to make full rank.

    Arguments:
        A: Directed network (np.ndarray, scipy.sparse.csr.csr_matrix)
        alpha: regularization term. Defaults to 0.

    Output:
        ranks: Solution to SpringRank
    """

    if alpha == 0:
        rank = csr_SpringRank(A)
    else:
        if type(A) == np.ndarray:
            A = scipy.sparse.csr_matrix(A)
        N = A.shape[0]
        k_in = scipy.sparse.csr_matrix.sum(A, 0)
        k_out = scipy.sparse.csr_matrix.sum(A, 1).T

        k_in = scipy.sparse.diags(np.array(k_in)[0], 0, [N, N], format="csr")
        k_out = scipy.sparse.diags(np.array(k_out)[0], 0, [N, N], format="csr")

        C = A + A.T
        D1 = k_in + k_out
        B = k_out - k_in
        B = B @ np.ones([N, 1])

        A = alpha * scipy.sparse.eye(N) + D1 - C
        rank = scipy.sparse.linalg.bicgstab(A, B)[0]

    return np.transpose(rank)

''' Credit to https://github.com/cdebacco/SpringRank/blob/master/python/tools.py for this function  '''
def shift_rank(ranks, episolon = 1e-10):
    '''
    Shifts all scores so that the minimum is in zero and the others are all positive
    '''
    min_r=min(ranks)-episolon
    N=len(ranks)
    for i in range(N): ranks[i]=ranks[i]-min_r
    return ranks    

def compute_predicted_ratings_spring_rank_old(games, pi_values):
    bin_data = binarize_data (games)
    unique_nodes = list(pi_values.keys())
    node_to_index = {node: index for index, node in enumerate(unique_nodes)}
    index_to_node = {v:k for k, v in node_to_index.items()}
    num_players = len(unique_nodes)

    A = np.full((num_players, num_players), 1.0, dtype=float)
    for i,j in bin_data:
        A[node_to_index[i]][node_to_index[j]] += 1


    shifted_ranks = shift_rank(get_ranks(A))
    pi_values = {index_to_node[index]: rank for index, rank in enumerate(shifted_ranks)}
    normalize_scores(pi_values)
    return pi_values

def compute_predicted_ratings_spring_rank_leadership_old(games, pi_values):
    bin_data = binarize_data_leadership (games)
    unique_nodes = list(pi_values.keys())
    node_to_index = {node: index for index, node in enumerate(unique_nodes)}
    index_to_node = {v:k for k, v in node_to_index.items()}
    num_players = len(unique_nodes)

    A = np.full((num_players, num_players), 1.0, dtype=float)
    for i,j in bin_data:
        A[node_to_index[i]][node_to_index[j]] += 1


    shifted_ranks = shift_rank(get_ranks(A))
    pi_values = {index_to_node[index]: rank for index, rank in enumerate(shifted_ranks)}
    normalize_scores(pi_values)
    return pi_values



@jit(nopython=True)
def eqs39(beta, s, A):
    N = A.shape[0]
    x = 0
    for i in range(N):
        for j in range(N):
            if A[i, j] == 0:
                continue
            else:
                x += (s[i] - s[j]) * (A[i, j] - (A[i, j] + A[j, i]) / (1 + np.exp(-2 * beta * (s[i] - s[j]))))
    return x


