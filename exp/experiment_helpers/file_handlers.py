
import os 
import csv

def save_results_to_csv(filename, headers, results):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data', filename)
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        for result in results:
            writer.writerow(result)


def save_instance_results(ho_likelihood, hol_likelihood, std_likelihood, base_dir, file_name): 


    results_dir = os.path.join(base_dir, 'data')
    file_path = os.path.join(results_dir, file_name)

    os.makedirs(results_dir, exist_ok=True)

    with open(file_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["ho_likelihood", "hol_likelihood", "std_likelihood"])
        for ho, hol, std in zip(ho_likelihood, hol_likelihood, std_likelihood):
            writer.writerow([ho, hol, std])


def read_files()