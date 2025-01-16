
import numpy as np 
from scipy.stats import spearmanr, kendalltau




def measure_rms(predicted_scores, ground_truth_scores, epsilon=1e-10):
    # Ensure the keys in both dictionaries are the same
    if predicted_scores.keys() != ground_truth_scores.keys():
        raise ValueError("Keys in predicted_scores and ground_truth_scores do not match.")
    

    predicted_scores = [score for player, score in sorted(predicted_scores.items(), key=lambda x: x[0])]
    true_scores = [score for player, score in sorted(ground_truth_scores.items(), key=lambda x: x[0])]

    predicted_scores = np.array(predicted_scores) + epsilon
    true_scores = np.array(true_scores) + epsilon


    rms = np.sqrt(np.mean((np.log(predicted_scores) - np.log(true_scores)) ** 2))
    
    return rms


def measure_rho(predicted_scores, ground_truth_scores):
    # measure predicted score of a player to it's true score

    predicted_scores = [score for player, score in sorted(predicted_scores.items(), key=lambda x: x[0])]
    true_scores = [score for player, score in sorted(ground_truth_scores.items(), key=lambda x: x[0])]

    correlation, p_val = spearmanr(predicted_scores, true_scores)

    return correlation


def measure_tau(predicted_scores, ground_truth_scores):
    predicted_scores = [score for player, score in sorted(predicted_scores.items(), key=lambda x: x[0])]
    true_scores = [score for player, score in sorted(ground_truth_scores.items(), key=lambda x: x[0])]

    # Calculate Kendall's tau correlation coefficient
    correlation, p_val = kendalltau(predicted_scores, true_scores)

    return correlation


def measure_log_likelihood (data, pi_values, model = 'ho_bt', epsilon=1e-10):
    
    
    log_like = log_prior = 0.0
    
    for i in pi_values:
        log_prior += np.log(pi_values[i]+epsilon) - 2.0 * np.log(1.0+pi_values[i]+epsilon)
        
    ############################################    
    if model == 'ho_bt':
        for i in range(0,len(data)):
            for j in range(0, len(data[i])-1):
                tmp = np.log(pi_values[data[i][j]] + epsilon)
                norm = 0.0
                for k in range(j, len(data[i])):
                    norm += pi_values[data[i][k]]
                tmp -= np.log(norm+epsilon)
                log_like += tmp
            
    ############################################    
    if model == 'ho_bt_leader':
        for i in range(0,len(data)):
            tmp = np.log(pi_values[data[i][0]] + epsilon)
            norm = 0.0
            for j in range(0, len(data[i])):
                norm += pi_values[data[i][j]]
            tmp -= np.log(norm+epsilon)
            log_like += tmp

    
    return log_like/len(data), log_prior
    
#measure in average log likelihood in accordance to a weighted test set
# def measure_likelihood(pred_ranking, testing_set, epsilon=1e-10):
#     total_log_likelihood = 0.0

#     for game in testing_set:
#         for j in range(len(game)-1):
#             total_ratings = sum(pred_ranking[k] for k in game[j:])
#             total_log_likelihood +=  np.log(pred_ranking[game[j]]) - np.log(total_ratings)


#     total_games = len(testing_set)
#     return total_log_likelihood / total_games   

def measure_likelihood(pred_ranking, testing_set, epsilon=1e-10):
    total_log_likelihood = 0.0

    for game in testing_set:

        tmp = sum(pred_ranking[k] for k in game)
        for i in range(len(game)-1):
            
            for j in range(i+1, len(game)):
                total_log_likelihood += np.log(pred_ranking[game[i]]+epsilon) - np.log(tmp+epsilon) 
            tmp -= pred_ranking[game[i]]

    total_games = len(testing_set)
    return total_log_likelihood / total_games   




def measure_leadership_likelihood(pred_ranking, testing_set, epsilon=1e-10):

    total_log_likelihood = 0.0

    for game in testing_set:
        tmp = np.log(pred_ranking[game[0]] + epsilon)
        total_ratings = sum(pred_ranking[k] for k in game)
        tmp -= np.log(total_ratings + epsilon)
        total_log_likelihood +=  tmp

    total_games = len(testing_set)
    return total_log_likelihood/total_games


