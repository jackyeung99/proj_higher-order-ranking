from scipy.stats import kendalltau
import numpy as np


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
        r1 = np.random.shuffle(list(range(100)))
        r2 = np.random.shuffle(list(range(100)))

        assert np.isclose(kendalltau(r1, r2)[0], 0)
