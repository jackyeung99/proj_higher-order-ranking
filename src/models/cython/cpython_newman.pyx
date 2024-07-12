import numpy as np
cimport numpy as np
from libc.math cimport exp, log

def normalize_scores(np.ndarray[double, ndim=1] scores):
    cdef int i
    cdef np.ndarray[double, ndim=1] scores_nonzero = scores[scores != 0]
    if len(scores_nonzero) == 0:
        return
    cdef double norm = exp(np.sum(np.log(scores_nonzero)) / len(scores_nonzero))
    for i in range(len(scores)):
        scores[i] /= norm

cpdef double iterate_equation_newman_weighted(int player_idx, np.ndarray[double, ndim=1] pi_values, list games_with_players):
    cdef double a = 1.0 / (pi_values[player_idx] + 1.0)
    cdef double b = 1.0 / (pi_values[player_idx] + 1.0)
    cdef int i, K, position, v, j
    cdef tuple game
    cdef double weight
    cdef np.ndarray[double, ndim=1] score_sums
    cdef np.ndarray[double, ndim=1] cumulative_sum
    cdef double tmp1, tmp2, tmp

    for K, position, game, weight in games_with_players:

        score_sums = np.array([pi_values[p] for p in game], dtype=np.float64)
        cumulative_sum = np.zeros(K + 1, dtype=np.float64)

        for j in range(1, K + 1):
            cumulative_sum[j] = cumulative_sum[j - 1] + score_sums[j - 1]

        if position < K - 1:
            tmp1 = cumulative_sum[K] - cumulative_sum[position + 1]
            tmp2 = tmp1 + score_sums[position]
            if tmp2 != 0:
                a += weight * (tmp1 / tmp2)

        for v in range(position):
            tmp = cumulative_sum[K] - cumulative_sum[v]
            if tmp != 0:
                b += weight * (1.0 / tmp)

    return a / b

cpdef double iterate_equation_newman_leadership_weighted(int player_idx, np.ndarray[double, ndim=1] pi_values, list games_with_players):
    cdef double a = 1.0 / (pi_values[player_idx] + 1.0)
    cdef double b = 1.0 / (pi_values[player_idx] + 1.0)
    cdef int i, K, position
    cdef tuple game
    cdef double weight
    cdef np.ndarray[double, ndim=1] score_sums
    cdef double tmp1, tmp2, tmp

    for i in range(len(games_with_players)):
        K, position, game, weight = games_with_players[i]
        score_sums = np.array([pi_values[game[p]] for p in range(K)], dtype=np.float64)
        
        if position == 0:
            tmp1 = np.sum(score_sums[1:K])
            tmp2 = np.sum(score_sums[0:K])
            if tmp2 != 0:
                a += weight * (tmp1 / tmp2)
        else:
            tmp = np.sum(score_sums[0:K])
            if tmp != 0:
                b += weight * (1.0 / tmp)

    return a / b

cpdef dict synch_solve_equations(dict bond_matrix, int max_iter, dict pi_values, object method, double sens=1e-10):
    cdef int s, iteration
    cdef double err
    cdef np.ndarray[double, ndim=1] scores
    cdef np.ndarray[double, ndim=1] tmp_scores
    cdef int player_idx
    cdef list games_with_player
    cdef list players = list(pi_values.keys())
    cdef Py_ssize_t num_players = len(players)
    
    scores = np.ones(num_players, dtype=np.float64)
    normalize_scores(scores)
   
    err = 1.0
    iteration = 0
    
    while iteration < max_iter and err > sens:
        err = 0
        tmp_scores = np.zeros(num_players, dtype=np.float64)

        for s in range(num_players):
            player_idx = players[s]
            if player_idx in bond_matrix:
                games_with_player = bond_matrix[player_idx]
                # Ensure each game entry is a list
                games_with_player = [list(game) if isinstance(game, tuple) else game for game in games_with_player]
                tmp_scores[s] = method(s, scores, games_with_player)
        
        normalize_scores(tmp_scores)
       
        err = np.max(np.abs(tmp_scores - scores))
        scores = tmp_scores.copy()
        
        iteration += 1
        
    final_scores = {players[i]: scores[i] for i in range(num_players)}
    return final_scores
