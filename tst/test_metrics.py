import os 
import sys
from scipy.stats import kendalltau
import numpy as np
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
sys.path.append(repo_root)

from src import *
from src.utils.metrics import *
from tst.tst_weight_conversion.old_newman import *

    
class TestMetrics:
    def test_kendall_tau_match(self):
        r1 = [1, 2, 3, 4]
        r2 = [1, 2, 3, 4]

        assert kendalltau(r1, r2)[0] == 1

    def test_kendall_tau_reverse(self):
        r1 = [1, 2, 3, 4]
        r2 = [4, 3, 2, 1]

        assert kendalltau(r1, r2)[0] == -1

    def test_kendall_tau_uncorrelated(self):
        r1 = list(range(100))
        r2 = list(range(100))

        np.random.shuffle(r1)
        np.random.shuffle(r2)

        correlation, _ = kendalltau(r1, r2)
        assert np.isclose(correlation, 0, atol=0.1)


    def test_rho(self):

        test_1 = {1:2.0, 2:1.0, 3:4.0}
        test_2 = {1:4.0, 2:1.0, 3:2.0}
        ground_truth = {1:1.0, 2: 0.0, 3:3.0}

        test_1_val = measure_rho(test_1, ground_truth)   
        test_2_val = measure_rho(test_2, ground_truth)

        assert test_1_val > test_2_val

    def test_tau(self):

        test_1 = {1:2.0, 2:1.0, 3:4.0}
        test_2 = {1:4.0, 2:1.0, 3:2.0}
        ground_truth = {1:1.0, 2: 0.0, 3:3.0}

        test_1_val = measure_tau(test_1, ground_truth)   
        test_2_val = measure_tau(test_2, ground_truth)

        assert test_1_val > test_2_val

    def test_weighted_log(self):

        data, pi_values = generate_leadership_model_instance(1000,1000,15,15)
        training_set, testing_set = train_test_split(data, train_size=.8, random_state=None)
        weighted_train = convert_games_to_dict(training_set)
        weighted_test = convert_games_to_dict(testing_set)
 
        #non-weighted test set
        std = compute_predicted_ratings_HO_BT_old(training_set, pi_values)

        # non weighted
        non_weighted_likelihood, _ = measure_log_likelihood(testing_set, std)
        non_weighted_leadership_likelihood, _ = measure_log_likelihood(testing_set, std, model='ho_bt_leader')
        avg_like = non_weighted_likelihood/len(testing_set)
        avg_lead_like = non_weighted_leadership_likelihood/len(testing_set)
 
        #weighted test set
        df = run_models_synthetic(weighted_train, weighted_test, pi_values)
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None):print(df)
        row = df[df['model'] == 'HO_BT']
        weighted_log_likelihood = row['log-likelihood'].values[0]
        weighted_leadership_likelihood = row['leadership-log-likelihood'].values[0]

        print(weighted_leadership_likelihood)
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
       