import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.graph_tools import *

def compute_point_wise_ratings(training_set, pi_values):
    scores = {k: 0 for k in pi_values.keys()}
    num_games = {k: 0 for k in pi_values.keys()}

    for game, weight in training_set.items():
        num_players = len(game)
        for idx, player in enumerate(game):
            num_games[player] += weight
            scores[player] += weight * ((num_players - idx) / num_players)

    final_scores = {player: (scores[player] / num_games[player]) if num_games[player] > 0 else 0 for player in scores}

    return final_scores

