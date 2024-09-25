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
        ''' test higher order, higher order newman, and std newman all reduce to the same predicted ratings in dyadic games'''
        data, pi_values = generate_model_instance(50, 50, 2, 2)
       
        data = convert_games_to_dict(data)
        # Get predictions from each model
        newman = get_predictions('BT', data, pi_values)
        newman_leadership = get_predictions('BT_leadership', data, pi_values)
        ho = get_predictions('HO_BT', data, pi_values)
        hol = get_predictions('HOL_BT', data, pi_values)
        
        # Assert that all prediction lists have the same length as pi_values
        predictions = [newman, newman_leadership, ho, hol]
        assert all(len(pred) == len(pi_values) for pred in predictions), "Length mismatch in predictions"
        
        
        # Check all combinations of models for closeness
        model_names = ['newman', 'newman_leadership', 'higher_order_newman', 'higher_order_leadership']
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                pred_i = predictions[i]
                pred_j = predictions[j]

                assert all(np.isclose(pred_i[key], pred_j[key], atol=1e-10) for key in pi_values.keys()), f"Predictions differ between {model_names[i]} and {model_names[j]}"



    def test_games_shuffled(self):
        ''' test that the ordering in which games are presented do not alter predicted ratings'''
         # Generate the model instance with specified parameters
        data, pi_values = generate_model_instance(50, 50, 2, 2)
        
        # Shuffle the data
        shuffled = data.copy()
        random.shuffle(shuffled)
        
        
        # Convert games data to dictionary format for predictions
        data_dict = convert_games_to_dict(data)
        shuffled_data_dict = convert_games_to_dict(shuffled)
        
        # Get predictions from the model
        newman = get_predictions('BT', data_dict, pi_values)
        shuffled_newman = get_predictions('BT', shuffled_data_dict, pi_values)
        
        # Check if the predictions are close enough
        assert all(np.isclose(shuffled_newman[key], newman[key], atol=1e-15) for key in pi_values.keys())


    def test_leadership_shuffled(self):
        ''' Check that leadership likelihood and likelihood output the same values for leadership diatic games, i.e. diadic leadership should be equivalent to diadic normal'''
        data, pi_values = generate_leadership_model_instance(50, 50, 2, 2)
         # Shuffle the data
        shuffled = data.copy()
        random.shuffle(shuffled)

        data_dict = convert_games_to_dict(data)
        shuffled_data_dict = convert_games_to_dict(shuffled)

        # Get predictions from each model
        newman = get_predictions('BT_leadership', data_dict, pi_values)
        shuffled_newman = get_predictions('BT_leadership', shuffled_data_dict, pi_values)
        
        # Check all combinations of models for closene
        assert all(np.isclose(shuffled_newman[key], newman[key], atol=1e-6) for key in pi_values.keys())

    def test_leadership_similarity(self): 
        ''' Check that leadership likelihood and likelihood output the same predicted ratings for synthetic diatic games'''
        models = [('BT', 'BT_leadership'), ('HO_BT', 'HOL_BT'), ('Spring_Rank', 'Spring_Rank_Leadership'), ('Page_Rank', 'Page_Rank_Leadership')] 

        for model, leadership in models: 

            data, pi_values = generate_model_instance(50, 50, 2, 2)
                
            data = convert_games_to_dict(data)

            model = get_predictions(model, data, pi_values)
            leadership = get_predictions(leadership, data, pi_values)

            assert all(np.isclose(leadership[key], model[key], atol=1e-15) for key in pi_values.keys())





