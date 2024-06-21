import os
import sys
import csv
import importlib.util
from sklearn.model_selection import ShuffleSplit

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.experiment_helpers.metrics import * 
from src.models.higher_order_newman import * 
from src.models.point_wise import *
from src.models.newman import * 
from src.experiment_helpers.file_handlers import * 


def get_predictions(model, training_set, pi_values):

    if model == 'newman':
        return compute_predicted_rankings_std(training_set, pi_values)
    elif model == 'newman_leadership':
        return compute_predicted_rankings_std_leadership(training_set, pi_values)
    elif model == 'higher_order_newman':
        return compute_predicted_rankings_std(training_set, pi_values)
    elif model == 'higher_order_leadership':
        return compute_predicted_rankings_std_leadership(training_set, pi_values)
    elif model == 'tensor_flow':
        pass
    elif model == 'point_wise':
        return compute_point_wise_ratings(training_set, pi_values)



def run_experiment(dataset_file_path, models, splits, metric='std_likelihood'):

    for file in os.listdir(dataset_file_path):

        if file.endswith('.soc'):
            file_path = os.path.join(dataset_file_path, file)
            data, pi_values = read_strict_ordered_dataset(file_path)

            if len(data) > 50: 
                shuffle_split = ShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)

                for idx, (train_index, test_index) in enumerate(shuffle_split.split(data)):

                    
                    train_data = [data[i] for i in train_index]
                    test_data = [data[i] for i in test_index]

                    likelihoods_dict = {'Game': list(range(len(test_data)))}

                    for model in models:
                        
                        predicted_rankings = get_predictions(model, train_data, pi_values)


                        if metric == 'std_likelihood':
                            game_likelihoods = compute_likelihood(predicted_rankings, test_data)
                        else:
                            game_likelihoods = compute_leadership_likelihood(predicted_rankings, test_data)


                        likelihoods_dict[model] = game_likelihoods

                    df = pd.DataFrame(likelihoods_dict)
                    split_filename = f'exp/ex03/raw_data/split_{idx + 1}_results.csv'
                    df.to_csv(split_filename, index=False)
            

                        
                    break



if __name__ == '__main__':
    # Run From ex03_realdata directory
    # rsync -zaP burrow:multi-reactive_rankings/higher_order_ranking/exp/ex03_realdata/data ~/senior_thesis/higher_order_ranking/exp/ex03_realdata/data


    dataset_file_path = 'datasets/preflib_datasets'
    models = ['newman', 'newman_leadership', 'higher_order_newman', 'higher_order_leadership', 'point_wise']
    splits = 10
    run_experiment(dataset_file_path, models, splits) 