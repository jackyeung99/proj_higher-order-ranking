import sys
import os 
import pytest


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.models.SpringRank import *
from src.utils import *
from tst.tst_weight_conversion.old_spring_rank import *
from tst.tst_weight_conversion.old_page_rank import *





def test_spring_rank():
    data, pi_values = generate_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_spr = get_predictions('spring_rank', weighted_data, pi_values)
    old_spr = compute_predicted_ratings_spring_rank_old(data, pi_values)

    assert len(old_spr) == len(new_spr)     
    assert all(np.isclose(old_spr[player], new_spr[player], atol=1e-10) for player in new_spr)


def test_spring_rank_leadership():
    data, pi_values = generate_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_spr = get_predictions('spring_rank_leadership', weighted_data, pi_values)
    old_spr = compute_predicted_ratings_spring_rank_leadership_old(data, pi_values)
    
    
    assert len(old_spr) == len(new_spr)
    assert all(np.isclose(old_spr[player], new_spr[player], atol=1e-10) for player in new_spr)


def test_page_rank():
    data, pi_values = generate_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_pr = get_predictions('page_rank', weighted_data, pi_values)
    old_pr = compute_predicted_ratings_page_rank_old(data, pi_values)

    assert len(old_pr) == len(new_pr)     
    assert all(np.isclose(old_pr[player], new_pr[player], atol=1e-10) for player in new_pr)

def test_page_rank_leadership():
    data, pi_values = generate_leadership_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_pr = get_predictions('page_rank_leadership', weighted_data, pi_values)
    old_pr = compute_predicted_ratings_page_rank_leadership_old(data, pi_values)

    assert len(old_pr) == len(new_pr)     
    assert all(np.isclose(old_pr[player], new_pr[player], atol=1e-10) for player in new_pr)