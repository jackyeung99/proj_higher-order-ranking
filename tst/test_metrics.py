import os 
import sys
from scipy.stats import kendalltau
import numpy as np
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
sys.path.append(repo_root)

from src import *

def measure_likelihood_old(pred_ranking, testing_set):
    return [recursive_probability_estimation_old(pred_ranking, game) for game in testing_set]


def recursive_probability_estimation_old(pred_rating, game, total_prob_estimation=0, episilon=1e-10):
    player_ratings = [pred_rating[player] for player in game]
    first_pos = player_ratings[0]
    total_ratings = sum(player_ratings)

    total_prob_estimation += np.log(first_pos+episilon) - np.log(total_ratings+episilon)

    if len(player_ratings) > 2:
        return recursive_probability_estimation_old(pred_rating, game[1:], total_prob_estimation)
    else:
        return total_prob_estimation 
    
        


class TestMetrics:
    def test_kendall_tau_match(self):
        r1 = [1, 2, 3, 4]
        r2 = [1, 2, 3, 4]

        assert kendalltau(r1, r2)[0] == 1

    def test_kendall_tau_reverse(self):
        r1 = [1, 2, 3, 4]
        r2 = [4, 3, 2, 1]

        assert kendalltau(r1, r2)[0] == -1

    # def test_kendall_tau_uncorrelated(self):
    #     r1 = np.random.shuffle(list(range(100)))
    #     r2 = np.random.shuffle(list(range(100)))

    #     assert np.isclose(kendalltau(r1, r2)[0], 0)


    def test_weighted_log(self):

        data, pi_values = generate_leadership_model_instance(1000,1000,2,2)
        training_set, testing_set = train_test_split(data, train_size=.8, random_state=None)
        weighted_train = convert_games_to_dict(training_set)
        weighted_test = convert_games_to_dict(testing_set)
    
        print(len(testing_set))
        #non-weighted test set
        std = compute_predicted_ratings_std(weighted_train, pi_values)

        old_likelihood = np.sum(measure_likelihood_old(std, testing_set))

        # non weighted
        non_weighted_likelihood, _ = measure_log_likelihood(testing_set, std)
        non_weighted_leadership_likelihood, _ = measure_log_likelihood(testing_set, std, model='ho_bt_leader')

        avg_like = non_weighted_likelihood/len(testing_set)
        avg_lead_like = non_weighted_leadership_likelihood/len(testing_set)
 
        #weighted test set
        df = run_models_synthetic(weighted_train, weighted_test, pi_values)
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None):print(df)
        row = df[df['model'] == 'newman']
        weighted_log_likelihood = row['log-likelihood'].values[0]
        weighted_leadership_likelihood = row['leadership-log-likelihood'].values[0]

        print(weighted_leadership_likelihood)
        assert np.isclose(old_likelihood, non_weighted_likelihood) 
        assert np.isclose(weighted_log_likelihood, avg_like)
        assert np.isclose(weighted_leadership_likelihood, avg_lead_like)

    def test_pr(self):

        data, pi_values = generate_model_instance(100,1000,4,4)
        training_set, testing_set = train_test_split(data, train_size=.8, random_state=None)
        weighted_train = convert_games_to_dict(training_set)
        weighted_test = convert_games_to_dict(testing_set)

        #non-weighted test set
        pr = compute_predicted_ratings_page_rank(weighted_train, pi_values)
 
        rms = measure_rms(pr, pi_values)
        rho = measure_rho(pr, pi_values)

        assert rms != 0
        assert rho != 1.0
       