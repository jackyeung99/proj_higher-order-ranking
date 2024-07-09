import sys
import os 
import pytest
import cProfile
import pstats

import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.models.newman import *
from src.utils import *
from datasets.utils.extract_ordered_games import *
from tst.tst_weight_conversion.old_newman import *


def profile_test_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()
    return profiler

def print_profile_stats(profiler, sort_by='cumtime', top_n=10):
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(sort_by)
    stats.print_stats(top_n)


def test_weighted_newman():
    data, pi_values = generate_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_newman = compute_predicted_ratings_std(weighted_data, pi_values)
    old_newman = compute_predicted_ratings_std_old(data, pi_values)
  
    
    assert len(old_newman) == len(new_newman)
    assert all(np.isclose(old_newman[player], new_newman[player], atol=1e-10) for player in new_newman)

def test_binarized_leadership(): 
    data, pi_values = generate_leadership_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_newman = compute_predicted_ratings_std_leadership(weighted_data, pi_values)
    old_newman = compute_predicted_ratings_std_leadership_old(data, pi_values)
  

    assert len(old_newman) == len(new_newman)
    assert all(np.isclose(old_newman[player], new_newman[player], atol=1e-10) for player in new_newman)

def test_weighted_ho():
    data, pi_values = generate_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_newman = compute_predicted_ratings_ho(weighted_data, pi_values)
    old_newman = compute_predicted_ratings_ho_old(data, pi_values)
  
    
    assert len(old_newman) == len(new_newman)
    assert all(np.isclose(old_newman[player], new_newman[player], atol=1e-10) for player in new_newman)



def test_weighted_hol():
    data, pi_values = generate_leadership_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_newman = compute_predicted_ratings_hol(weighted_data, pi_values)
    old_newman = compute_predicted_ratings_hol_old(data, pi_values)
  
   
    assert len(old_newman) == len(new_newman)
    assert all(np.isclose(old_newman[player], new_newman[player], atol=1e-10) for player in new_newman)



def test_speed_gains():
    data, pi_values = generate_model_instance(1000, 1000, 10, 10)
    weighted_data = convert_games_to_dict( data)
    
    start = time.perf_counter()
    
    # compute_predicted_ratings_std(weighted_data, pi_values)
    # compute_predicted_ratings_std_leadership(weighted_data, pi_values)
    compute_predicted_ratings_ho(weighted_data, pi_values)
    # compute_predicted_ratings_hol(weighted_data, pi_values)
    end = time.perf_counter()
    weighted_speed = end - start
    print(f"Weighted + numba speed: {weighted_speed}")

    
    start = time.perf_counter()
   
    compute_predicted_ratings_std_old(data, pi_values)
    # compute_predicted_ratings_std_leadership_old(data, pi_values)
    compute_predicted_ratings_ho_old(data, pi_values)
    # compute_predicted_ratings_hol_old(data, pi_values)
    end = time.perf_counter()
    standard_speed = end - start
    print(f"Standard speed: {standard_speed}")

    assert weighted_speed < standard_speed

def main():
    data, pi_values = generate_model_instance(1000, 1000, 8, 8)
    weighted_data = convert_games_to_dict(data)
    
    profiler = profile_test_function(compute_predicted_ratings_ho, weighted_data, pi_values)
    print_profile_stats(profiler)

    profiler = profile_test_function(compute_predicted_ratings_ho_old, data, pi_values)
    print_profile_stats(profiler)

if __name__ == '__main__':
    main()