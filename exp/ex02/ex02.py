import os
import sys
import logging
from sklearn.model_selection import ShuffleSplit

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.experiment_helpers import *


FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate_model_prediction(N, M, K1, K2, splits):

    pi_values, data = generate_model_instance(N, M, K1, K2)
    random.shuffle(data)

    shuffle_split = ShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)

    for idx, (train_index, test_index) in enumerate(shuffle_split.split(data)):
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        file_name = os.path.join(repo_root, f"exp/ex02/data/N-{N}_M-{M}_K1-{K1}_K2-{K2}_rep-{idx}.csv")
        run_models(train_data, test_data, pi_values, file_name)

        

if __name__ == '__main__':

    # rsync -zaP burrow:multi-reactive_rankings/higher_order_ranking/exp/ex02_predictive_newman/data ~/senior_thesis/higher_order_ranking/exp/ex02_predictive_newman
    # parallel --jobs 24 python3 ex02.py {1} {2} {3} ::: 1000 ::: 64 141 312 689 1521 3360 7419 16384 ::: $(seq 2 10)
    N, M, K1 = map(int, sys.argv[1:])
    
    K2 = K1 + 1
    logging.debug(f'running code for {N} {M} {K1} {K2}')
    evaluate_model_prediction(N, M, K1, K2, 20)
        