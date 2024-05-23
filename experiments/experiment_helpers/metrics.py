import math 
import numpy as np 
from scipy import stats
from scipy.stats import spearmanr, kendalltau

# helpers
def get_rankings(scores):
    # Sort the dictionary by scores in descending order
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    # Extract the player names to get the rankings
    rankings = [player for player, score in sorted_scores]
    return rankings

def get_sorted_player_scores(scores):
    # Sort the dictionary by player 
    sorted_players = sorted(scores.items(), key=lambda item: item[0])
    # Extract the player scores to get the rankings
    player_names = [score for player, score in sorted_players]
    return player_names



def calculate_rms(predicted_scores, ground_truth_scores):
    # predicted_scores = get_sorted_player_scores(predicted_scores)
    # true_scores = get_sorted_player_scores(ground_truth_scores)

    predicted_scores = [score for player, score in sorted(predicted_scores.items())]
    true_scores = [score for player, score in sorted(ground_truth_scores.items())]


    # Calculate the root mean square error
    return np.sqrt(np.mean( (np.log(np.array(predicted_scores)) - np.log(np.array(true_scores))) ** 2) )


def calculate_rho(predicted_scores, ground_truth_scores):
    # measure predicted score of a player to it's true score
    predicted_scores = get_sorted_player_scores(predicted_scores)
    true_scores = get_sorted_player_scores(ground_truth_scores)

    correlation, p_val = spearmanr(predicted_scores, true_scores)
    return correlation


def calculate_tau(predicted_scores, ground_truth_scores):
    # Sort both dictionaries by keys (players) to ensure alignment
    sorted_predicted = [score for player, score in sorted(predicted_scores.items())]
    sorted_true = [score for player, score in sorted(ground_truth_scores.items())]

    # Calculate Kendall's tau correlation coefficient
    correlation, p_val = kendalltau(sorted_predicted, sorted_true)
    return correlation


def measure_log_likelihood (data, pi_values, model = 'ho_bt'):
    
    
    log_like = log_prior = 0.0
    
    for i in pi_values:
        log_prior += np.log(pi_values[i]) - 2.0 * np.log(1.0+pi_values[i])
        
    ############################################    
    if model == 'ho_bt':
        for i in range(0,len(data)):
            for j in range(0, len(data[i])-1):
                tmp = np.log(pi_values[data[i][j]])
                norm = 0.0
                for k in range(j, len(data[i])):
                    norm += pi_values[data[i][k]]
                tmp -= np.log(norm)
            log_like += tmp
    ############################################    
    if model == 'ho_bt_leader':
        for i in range(0,len(data)):
            tmp = np.log(pi_values[data[i][0]])
            norm = 0.0
            for j in range(0, len(data[i])):
                norm += pi_values[data[i][j]]
            tmp -= np.log(norm)
            log_like += tmp
    #############################################
    if model == 'std_bt':
        for i in range(0,len(data)):
            for j in range(0, len(data[i])-1):
                for k in range(j+1, len(data[i])):
                    norm = pi_values[data[i][j]] + pi_values[data[i][k]]
                    tmp  = np.log(pi_values[data[i][j]]) - np.log(norm)
            log_like += tmp
    #############################################
    
    
    return log_like, log_prior

    


if __name__ == '__main__':
    predicted_scores = {
        'player1': 0.8,
        'player2': 0.6,
        'player3': 0.9,
        'player4': 0.5
    }

    ground_truth_scores = {
        'player1': 0.7,
        'player2': 0.8,
        'player3': 0.6,
        'player4': 0.9
    }


    print(get_rankings(ground_truth_scores))
    print(get_sorted_player_scores(ground_truth_scores))
    # Calculate metrics
    rms_error = calculate_rms(predicted_scores, ground_truth_scores)
    spearman_corr, _ = calculate_rho(predicted_scores, ground_truth_scores)
    kendall_corr, _ = calculate_tau(predicted_scores, ground_truth_scores)

    print("Root Mean Square Error:", rms_error)
    print("Spearman's Correlation:", spearman_corr)
    print("Kendall's Tau:", kendall_corr)