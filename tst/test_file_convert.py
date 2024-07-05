import sys
import os
import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_root)

from datasets.utils.convert_raw_files import *
from src import *

class TestSocCombination:

    @pytest.fixture
    def setup(self):
        data, pi_values = read_edge_list('tst_combine_edges.txt')
        return data, pi_values

    def test_len(self, setup):
        data, pi_values = setup
        assert len(data) == 3
        assert isinstance(data, dict)
        assert len(pi_values) == 5
    

    