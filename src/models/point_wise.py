
''' Average position '''
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

