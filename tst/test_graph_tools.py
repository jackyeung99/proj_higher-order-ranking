import os 
import sys

import numpy as np 
import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
sys.path.append(repo_root)

from src.utils import *


def binarize_data (data):

    bin_data = []

    for i in range(0, len(data)):

        K = len(data[i])
        for r in range(0, K-1):
            for s in range (r+1, K):
                bin_data.append([data[i][r],data[i][s]])


    return bin_data

def binarize_data_np(data):
    bin_data = []
    for arr in data:
        arr = np.array(arr)
        idx = np.triu_indices(len(arr), k=1)
        pairs = np.array([arr[idx[0]], arr[idx[1]]]).T
        bin_data.extend(pairs.tolist())
    return bin_data

def binarize_data_leadership (data):
    
    bin_data = []
    
    for i in range(0, len(data)):
        
        K = len(data[i])
        for s in range (1, K):
            bin_data.append([data[i][0],data[i][s]])
        
        
    return bin_data


def binarize_data_leadership_np(data):
    bin_data = []
    
    for arr in data:
        arr = np.array(arr)
        pairs = np.column_stack((np.repeat(arr[0], len(arr) - 1), arr[1:]))
        bin_data.extend(pairs.tolist())
        
    return bin_data


class TestGraphTools:

    def test_numpy_binarize(self):

        games = [(1,2,3), (4,5,6)]
        correct = [[1,2], [1,3], [2,3], [4,5], [4,6], [5,6]]

        np_binarize = binarize_data_np(games)
        bin = binarize_data(games)

        assert bin == np_binarize
        assert sorted(np_binarize) == sorted(correct)
        assert sorted(bin) == sorted(correct)

    def test_leadership_binarize(self): 
        games = [(1,2,3), (4,5,6)]
        correct = [[1,2], [1,3], [4,5], [4,6]]

        np_binarize = binarize_data_leadership_np(games)
        bin = binarize_data_leadership(games)

        assert bin == np_binarize
        assert sorted(np_binarize) == sorted(correct)
        assert sorted(bin) == sorted(correct)

    def test_binarize_diactic(self):

        games = [(1,2), (3,4), (5,6)]

        bin = binarize_data_np(games)
        binl = binarize_data_leadership(games)

        print(bin)
        print(binl)

        assert bin == binl


    def test_real_data(self):

        data, pi_values = generate_model_instance(50, 50, 3,3)
        
        assert len(binarize_data_np(data)) == 150
        assert len(binarize_data_leadership(data)) == 100

