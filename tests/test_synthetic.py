import numpy as np
from scipy.stats import logistic

N = 10000


def sample_pi():
    pi_values = {n: float(np.exp(logistic.rvs(size=1)[0])) for n in range(N)}

    return pi_values


def normalize_scores(pi_values):
    norm = 0.0
    val = 0.0
    for n in pi_values:
        norm = norm + np.log(pi_values[n])
        val = val + 1.0

    norm = np.exp(norm / val)

    for n in pi_values:
        pi_values[n] = pi_values[n] / norm

    return pi_values


class TestScoreSampling:
    def test_sample_pi(self):
        pi = sample_pi()
        assert len(pi) == N
        assert isinstance(pi, dict)
        assert np.all([v > 0 for v in pi.values()])

    def test_normalize_scores(self):
        pi = sample_pi()
        pi_normalized = list(normalize_scores(pi).values())

        assert np.isclose(np.prod(pi_normalized), 1)
