import os 
import sys

import argparse

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_root)

from src.utils.file_handlers import group_dataset_files
from src.utils.c_operation_helpers import run_simulation



if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run simulation convergence with specified dataset.")
    parser.add_argument(
        '--dset', type=str, required=True,
        help="Dataset number to be used (e.g., '00010')."
    )
    parser.add_argument(
        '--is_synthetic', type=int, choices=[0, 1], required=True,
        help="Specify whether the dataset is synthetic (1) or real (0)."
    )

    args = parser.parse_args()

    DATASET_PATH = os.path.join(repo_root, 'datasets')

    # Define paths based on dataset number
    dataset_folder = 'Synthetic_Data' if args.is_synthetic else 'Real_Data'
    node_path = os.path.join(DATASET_PATH, dataset_folder, f"{args.dset}_nodes.txt")
    edge_path = os.path.join(DATASET_PATH, dataset_folder, f"{args.dset}_edges.txt")

    # Run the simulation convergence
    results = run_simulation(node_path, edge_path, is_synthetic=args.is_synthetic)
    print(results)


