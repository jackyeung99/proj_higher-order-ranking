import os
import sys
import csv

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)


from src.solvers import *
from src.syntetic import *
from src.file_readers import * 
from exp.experiment_helpers.metrics import * 
from exp.experiment_helpers.model_evaluation import * 
from exp.experiment_helpers.file_handlers import * 


FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate_model_prediction(dataset, pi_values, data):

    for rep in range(50):

        random.shuffle(data)
        training_set, testing_set = split_games(data, .8)

        for model in ['ho', 'hol']:
                
            if model == 'ho':
                ho_likelihood, hol_likelihood, std_likelihood = benchmark_ho(training_set, testing_set, pi_values)
            else:
                ho_likelihood, hol_likelihood, std_likelihood = benchmark_hol(training_set, testing_set, pi_values)
            
            file_name = f"dataset-{dataset}_rep-{rep}_model-{model}.csv"
            save_instance_results( ho_likelihood, hol_likelihood, std_likelihood, FILE_DIR, file_name)



def run_experiments():
    def read_and_evaluate(file_path, read_data_function, data_label):
        data, pi_values = read_data_function(file_path)
        evaluate_model_prediction(data_label, pi_values, data)

    
    read_and_evaluate('data/fifa_wc.txt', read_data_fifa, 'fifa')
    read_and_evaluate('data/authorships.txt', read_data_authors, 'authors')
    read_and_evaluate('data/cl_data.txt', read_data_ucl, 'ucl')


if __name__ == '__main__':

    run_experiments() 