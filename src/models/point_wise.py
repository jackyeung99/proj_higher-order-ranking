import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.models.newman import *
from src.experiment_helpers.file_handlers import *
from src.experiment_helpers.metrics import * 
from src.experiment_helpers.synthetic import *
from src.utils import *

def compute_point_wise_ratings(training_set, pi_values):
    scores = {k: 0 for k in pi_values.keys()}
    num_games = {k: 0 for k in pi_values.keys()}

    for game in training_set:
        num_players = len(game)
        for idx, player in enumerate(game):
            num_games[player] += 1
            scores[player] += (num_players - idx) / num_players

    final_scores = {player: (scores[player] / num_games[player]) if num_games[player] > 0 else 0 for player in scores}

    return final_scores

if __name__ == '__main__':
    

    data, pi_values = read_strict_ordered_dataset('datasets/preflib_datasets/00004-00000001.soc')

    print(compute_point_wise_ratings(data, pi_values))