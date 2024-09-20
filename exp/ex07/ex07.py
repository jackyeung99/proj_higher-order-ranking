import os
import sys


from concurrent.futures import ProcessPoolExecutor

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.convergence_test_helpers import * 
from src.utils.graph_tools import generate_model_instance


def average_convergence(repetitions, out_file_dir):
    os.makedirs(out_dir, exist_ok=True)
    futures = []
    with ProcessPoolExecutor(max_workers=32) as executor:
         for rep in range(repetitions):
            data, pi_values = generate_model_instance(1000, 10000, 5, 5)
            file_name = os.path.join(out_file_dir, f"rep-{rep}.csv")
            futures.append(executor.submit(save_convergence_data, file_name, data, pi_values))


     


if __name__ == '__main__':

    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    average_convergence(100, out_file_dir=out_dir)






