import os 
import sys

import numpy as np 
import pytest
import random
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
sys.path.append(repo_root)

from src.models import *
from src.utils import *




class TestModels:


    def test_bin_games(self):
        ''' test higher order, higher order newman, and std newman all reduce to the same predicted ratings in diadic games'''
        data, pi_values = generate_model_instance(50, 50, 2, 2)
        
        
        # Get predictions from each model
        newman = get_predictions('newman', data, pi_values)
        newman_leadership = get_predictions('newman_leadership', data, pi_values)
        ho = get_predictions('higher_order_newman', data, pi_values)
        hol = get_predictions('higher_order_leadership', data, pi_values)
        
        # Assert that all prediction lists have the same length as pi_values
        predictions = [newman, newman_leadership, ho, hol]
        assert all(len(pred) == len(pi_values) for pred in predictions), "Length mismatch in predictions"
        
        # Check all combinations of models for closeness
        model_names = ['newman', 'newman_leadership', 'higher_order_newman', 'higher_order_leadership']
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                pred_i = predictions[i]
                pred_j = predictions[j]
                assert all(np.isclose(pred_i[key], pred_j[key], atol=1e-15) for key in pi_values.keys()), f"Predictions differ between {model_names[i]} and {model_names[j]}"



    def test_games_shuffled(self):
        ''' test that the ordering in which games are presented do not alter predicted ratings'''
        data, pi_values = generate_model_instance(50, 50, 2, 2)
        
        # Get predictions from each model
        newman = get_predictions('newman', data, pi_values)
        shuffled = random.shuffle(data)
        shuffled_newman = get_predictions('newman', data, pi_values)
        
        # Check all combinations of models for closene
        assert all(np.isclose(shuffled_newman[key], newman[key], atol=1e-15) for key in pi_values.keys())


    def test_leadership_shuffled(self):
        ''' Check that leadership likelihood and likelihood output the same values for leadership diatic games, i.e. diadic leadership should be equivalent to diadic normal'''
        data, pi_values = generate_leadership_model_instance(50, 50, 2, 2)
    
        # Get predictions from each model
        newman = get_predictions('newman', data, pi_values)
        shuffled = random.shuffle(data)
        shuffled_newman = get_predictions('newman', data, pi_values)
        
        # Check all combinations of models for closene
        assert all(np.isclose(shuffled_newman[key], newman[key], atol=1e-15) for key in pi_values.keys())

    def test_leadership_similarity(self): 
        ''' Check that leadership likelihood and likelihood output the same predicted ratings for synthetic diatic games'''
        models = [('newman', 'newman_leadership'), ('higher_order_newman', 'higher_order_leadership'), ('spring_rank', 'spring_rank_leadership'), ('page_rank', 'page_rank_leadership')] 

        for model, leadership in models: 

            data, pi_values = generate_model_instance(50, 50, 2, 2)

            model = get_predictions(model, data, pi_values)
            leadership = get_predictions(leadership, data, pi_values)

            assert all(np.isclose(leadership[key], model[key], atol=1e-15) for key in pi_values.keys())

    def test_likelihood(self): 
        ''' Check that leadership and normal models perform the same on diatic edges'''
        models = [('newman', 'newman_leadership'), ('higher_order_newman', 'higher_order_leadership'), ('spring_rank', 'spring_rank_leadership'), ('page_rank', 'page_rank_leadership')] 

        for model, leadership in models: 

            data, pi_values = generate_model_instance(50, 50, 2, 2)

            training_set, testing_set = train_test_split(data, train_size=.8, random_state=None)

            model = get_predictions(model, training_set, pi_values)
            leadership = get_predictions(leadership, training_set, pi_values)

            model_likelihood = compute_likelihood(model, testing_set)
            leadership_likelihood = compute_likelihood(leadership, testing_set)

            assert all(np.isclose(model_likelihood[game], leadership_likelihood[game], atol=1e-15) for game in range(len(testing_set)))

    def test_leadership_likelihood(self): 

        ''' Check that leadership likelihood and likelihood output the same values for diatic games'''

        data, pi_values = generate_model_instance(50, 50, 2, 2)

        training_set, testing_set = train_test_split(data, train_size=.8, random_state=None)

        predictions = get_predictions('newman', training_set, pi_values)

        likelihood = compute_likelihood(predictions, testing_set)
        leadership_likelihood = compute_leadership_likelihood(predictions, testing_set)

        assert all(np.isclose(likelihood[game], leadership_likelihood[game], atol=1e-15) for game in range(len(testing_set)))









