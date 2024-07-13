import os 
import sys
from scipy.stats import kendalltau
import numpy as np
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
sys.path.append(repo_root)

from src import *

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


    def test_log_likelihood(self):

        data, pi_values = generate_leadership_model_instance(100,100,4,4)

        training_set, testing_set = train_test_split(data, train_size=.8, random_state=None)
        weighted_train = convert_games_to_dict(training_set)
        
        stdl = compute_predicted_ratings_std_leadership(weighted_train, pi_values)
        hol = compute_predicted_ratings_hol(weighted_train, pi_values)

        std_likelihood = np.array(compute_leadership_likelihood(stdl, testing_set))
        hol_likelihood = np.array(compute_leadership_likelihood(hol, testing_set))

        new_stdl_likelihood, _ = measure_log_likelihood(testing_set, stdl, model='ho_bt_leader')
        new_hol_likelihood, _ = measure_log_likelihood(testing_set, hol, model='ho_bt_leader')

        assert np.isclose(new_stdl_likelihood, np.sum(std_likelihood), atol=1e-10)
        assert np.isclose(new_hol_likelihood, np.sum(hol_likelihood), atol=1e-10)