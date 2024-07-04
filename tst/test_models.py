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
        pi_values, data = generate_model_instance(50, 50, 2, 2)
        
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
        pi_values, data = generate_model_instance(50, 50, 2, 2)
        
        # Get predictions from each model
        newman = get_predictions('newman', data, pi_values)
        shuffled = random.shuffle(data)
        shuffled_newman = get_predictions('newman', data, pi_values)
        
        # Check all combinations of models for closene
        assert all(np.isclose(shuffled_newman[key], newman[key], atol=1e-15) for key in pi_values.keys())


    def test_leadership_shuffled(self):
        pi_values, data = generate_leadership_model_instance(50, 50, 2, 2)
    
        # Get predictions from each model
        newman = get_predictions('newman', data, pi_values)
        shuffled = random.shuffle(data)
        shuffled_newman = get_predictions('newman', data, pi_values)
        
        # Check all combinations of models for closene
        assert all(np.isclose(shuffled_newman[key], newman[key], atol=1e-15) for key in pi_values.keys())

    def test_leadership_similarity(self): 
        models = [('newman', 'newman_leadership'), ('higher_order_newman', 'higher_order_leadership'), ('spring_rank', 'spring_rank_leadership'), ('page_rank', 'page_rank_leadership')] 

        for model, leadership in models: 

            pi_values, data = generate_model_instance(50, 50, 2, 2)

            model = get_predictions(model, data, pi_values)
            leadership = get_predictions(leadership, data, pi_values)

            assert all(np.isclose(leadership[key], model[key], atol=1e-15) for key in pi_values.keys())

    def test_leadership_similarity_normal(self): 
        models = [('newman', 'newman_leadership'), ('higher_order_newman', 'higher_order_leadership'), ('spring_rank', 'spring_rank_leadership'), ('page_rank', 'page_rank_leadership')] 

        for model, leadership in models: 

            pi_values, data = generate_model_instance(50, 50, 2, 2)

            training_set, testing_set = train_test_split(data, train_size=.8, random_state=None)

            model = get_predictions(model, data, pi_values)
            leadership = get_predictions(leadership, data, pi_values)

            model_likelihood = get

            assert all(np.isclose(leadership[key], model[key], atol=1e-15) for key in pi_values.keys())








