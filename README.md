
# Efficient inference of rankings from multi-body comparisons

This project provides source code for the Efficient inference of rankings from multi-body comparisons. The repository also contains the original scientific analyses developed by the Authors (see below) for the paper

- **(Under review)** Yeung _et al_. 2025. [_Efficient inference of rankings from multi-body comparisons_](arxiv.org).

If you use this codebase, please cite the following:
- PAPER CITATION
- REPO CITATION

# Contents

- [Efficient inference of rankings from multi-body comparisons](#Efficient-inference-of-rankings-from-multi-body-comparisons)
- [Contents](#contents)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installing](#installing)
  - [Quick Start](#quick-start)
- [Usage](#usage)
  - [Reproducing experiments](#reproducing-experiments)
  - [Package Structure](#package-structure)
- [Documentation](#documentation)
- [Tests](#tests)
- [Other Information](#other-information)
  - [Built With](#built-with)
  - [Contributing](#contributing)
  - [Versioning](#versioning)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)


# Getting Started

The code base for this project is written in Python with package management handled with Conda.

These instructions will give you a copy of the project up and running on
your local machine for development, testing, and analysis purposes.

## Prerequisites

A compatible Python install is needed to begin - the package management is handled by Conda as described below.
- [Python \[3.10+\]](https://python.org/downloads/)
- [GNU Make \[4.2+\]](https://www.gnu.org/software/make/)

A complete list of utilized packages is available in the `requirements.txt` file. There is, however, a package dependency hierarchy where some packages in the `requirements.txt` are not strictly necessary for the utilization of package infrastructure. The core requirements are listed as dependencies in the build instructions. Further instructions for creating a controlled environment from this manifest is available below, in the [Installing](#installing) section.

<!-- > _Note: We personally recommend using mambaforge, an extension of conda that is considerably faster and more robust. Further information can be found in the [Mamba docs](https://mamba.readthedocs.io/en/latest/index.html)_. -->

## Installing

To (locally) reproduce this project, do the following:

1. Download this code base. Notice that raw data are typically not included in the git-history and may need to be downloaded independently - see [Reproducing Experiments](#reproducing-experiments) for more information.
2. (Optional) Open a terminal with Python installed and create a new virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install the package
   ```bash
   pip install .
   ```


This will install all necessary packages for you to be able to run the scripts and everything should work out of the box.

## Quick Start
This guide provides simple instructions for running the simulation on a chosen dataset.

1. Compile source code

The core of the ranking calculations are written in an efficient C implementation. This must be compiled before the python scripts running the experiments will work. You can use the provided `makefile` in the `C_prog/` directory to compile this for UNIX-based machines out-of-the-box; Windows machines will need to edit some of the compiler flags within the `makefile`. The compilation can be accomplished by running the following from the root directory

```bash
cd C_prog
make
cd ..
```

2. Locate the Dataset

Find the ID of the dataset you want to use by checking the datasets/dataset.info file. Each dataset is assigned a unique ID that must be formatted as a 5-digit number, with leading zeroes if necessary (e.g., 00001). Ensure that the selected dataset has a file for both its edges and nodes within dataset/Real_Data/

3. Run the Model

if the dataset has the true scores set is_synthetic = 1 otherwise is_synthetic = 0

Run the model on the selected dataset using the following command:

```bash
python3 src/test.py --dataset_number=00001 --is_synthetic=0
```


# Usage
## Reproducing experiments


### Synthetic

to generate synthetic results(note that this will create a large amount of files)

```bash
python3 datasets/utils/gen_synthetic_data 
```

#### Accuracy   
To run the experiments on the accuracy of all four models on the synthetic data
```bash
python3 exp/ex01/ex01 
```

This will download each result into the folder exp/ex01/data 
to preprocess and visualize these result run all cells within notebook/ex01_synthetic_accuracy  

#### Convergence  
To run the experiments on the convergence of our model and zermellos 
```bash
python3 exp/ex02/ex02
```
This will result in a table being saved into the folder exp/ex02/results
to visualize these results run all cells within the file notebook/ex02_synthetic_convergence

### Real Results
To run all datasets included in the paper 

#### Accuracy   
To run the experiments on the accuracy of all four models on the synthetic data
```bash
python3 exp/ex03/ex03 
```

This will download each result into the folder exp/ex03/data 
to preprocess and visualize these result run all cells within notebook/ex01_real_accuracy  

#### Convergence  
To run the experiments on the convergence of our model and zermellos 
```bash
python3 exp/ex04/ex04
```
This will result in a table being saved into the folder exp/ex04/results
to visualize these results run all cells within the file notebook/ex04_real_convergence



## Package Structure

```text
├── C_Prog                         * Efficient C implementation 
│   ├── Convergence_Readfile       * Measure convergence results
│   │   ├── bt_functions.c
│   │   ├── bt_functions.h
│   │   ├── bt_model_data.c
│   │   ├── bt_model_data.out
│   │   ├── makefile
│   │   ├── mt19937-64.c
│   │   ├── mt64.h
│   │   ├── my_sort.c
│   │   └── my_sort.h
│   └── Readfile                   * Measure accuracy
│       ├── bt_functions.c
│       ├── bt_functions.h
│       ├── bt_model_data.c
│       ├── bt_model_data.out
│       ├── makefile
│       ├── mt19937-64.c
│       ├── mt64.h
│       ├── my_sort.c
│       └── my_sort.h
├── LICENSE
├── README.md
├── datasets
│   ├── Real_Data                    * Edges and Nodes of datasets used in paper 
│   │   ├── 00001_edges.txt
│   │   ├── 00001_nodes.txt
│   │   ├── 00002_edges.txt
│   │   ├── 00002_nodes.txt
│   │   ├── 00003_edges.txt
│   │   ├── 00003_nodes.txt
│   │   ├── 00004_edges.txt
│   │   ├── 00004_nodes.txt
│   │   ├── 00005_edges.txt
│   │   ├── 00005_nodes.txt
│   │   ├── 00006_edges.txt
│   │   ├── 00006_nodes.txt
│   │   ├── 00007_edges.txt
│   │   ├── 00007_nodes.txt
│   │   ├── 00008_edges.txt
│   │   ├── 00008_nodes.txt
│   │   ├── 00009_edges.txt
│   │   └── 00009_nodes.txt
│   ├── dataset_info.csv             * information on edge size, number of players, number of games, and mappings of dataset names and ids 
│   └── utils                        * preprocessing 
│       ├── convert_raw_files.py
│       ├── dataset_info.py
│       ├── extract_ordered_games.py
│       ├── gen_synthetic_data.py
│       └── rename_datasets.py
├── doc
│   ├── experiment_descriptions.txt
│   └── sketch_experiment.txt
├── exp
│   ├── ex01
│   │   ├── ex01.py
│   │   └── results
│   │       ├── leadership_log_likelihood_summary.csv
│   │       ├── log_likelihood_summary.csv
│   │       ├── rho_summary.csv
│   │       └── tau_summary.csv
│   ├── ex02
│   │   ├── ex02.py
│   │   └── results
│   │       └── Convergence_Table.csv
│   ├── ex03
│   │   ├── ex03.py
│   │   └── results
│   │       ├── leadership_log_likelihood_summary.csv
│   │       └── log_likelihood_summary.csv
│   └── ex04
│       ├── ex04.py
│       └── results
│           └── Convergence_Table.csv
├── notebook
│   ├── comparison_models.ipynb
│   ├── convergence_behavior.ipynb
│   ├── ex01_synthetic_accuracy.ipynb
│   ├── ex02_synthetic_convergence.ipynb
│   ├── ex03_real_accuracy.ipynb
│   ├── ex04_real_convergence.ipynb
│   ├── figure_settings
│   │   ├── __init__.py
│   │   ├── ieee.mplstyle
│   │   ├── science.mplstyle
│   │   └── settings.py
│   └── training_size.ipynb
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── archive
│   │   ├── weighted_bt.py
│   │   └── weighted_graph_helpers.py
│   ├── models                  * All models including comparisons to Zermello and other graph ranking algorithms 
│   │   ├── BradleyTerry.py     * Python representation of our model
│   │   ├── SpringRank.py
│   │   ├── __init__.py
│   │   ├── page_rank.py
│   │   ├── point_wise.py
│   │   └── zermello.py
│   |── utils
│   |   ├── __init__.py
│   |   ├── c_operation_helpers.py          * run c code 
│   |   ├── convergence_test_helpers.py      
│   |   ├── file_handlers.py                
│   |   ├── graph_tools.py                  * building hypergraphs
│   |   ├── metrics.py                     
│   |   └── operation_helpers.py            * run python implmentations
|   |_ test.py                              * Example Run
|
|
└── tst
    ├── test_graph_tools.py
    ├── test_metrics.py
    ├── test_models.py
    ├── test_operation_helpers.py
    └── test_synthetic.py
```


# Documentation

This repository does not maintain extensive independent documentation for its source code. We do, however, include documentation and notes on scientific experiments we've conducted throughout the project. If you are interested in seeing these notes, please email [Filippo Radicchi](mailto:filrad@iu.edu) with your inquiry.

We have, however, kept all experimental protocols related to the final experimental designs of the published results in this public repository. These can be found in `docs/experiments/` with the appropriate names matching the results as presented in the manuscript.

<!-- Additionally, a copy of individual derivations can be found in `docs/` that are highly suggestive of methodological choices and implications for our work. -->

# Tests

All unit tests are written with [pytest](docs.pytest.org).

Tests can be run directly with the commands:
```bash
pip install pytest
pytest tests/
```



# Other Information
## Built With
  - [ChooseALicense](https://choosealicense.com/) - Used to choose
    the license

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/jackyeung99/higher_order_ranking/tags).

## Authors

All correspondence shoulld be directed to [Filippo Radicchi](mailto:filrad@iu.edu).

- Jack Yeung
- Daniel Kaiser
- Filippo Radicchi

## License

This project is licensed under the [MIT License](LICENSE.md)
Creative Commons License - see the [LICENSE](LICENSE.md) file for
details.

## Acknowledgments
  - **Billie Thompson** - *Provided README and CONTRIBUTING template* -
  [PurpleBooth](https://github.com/PurpleBooth)
  - **George Datseris** - *Published workshop on scientific code; inspired organization for reproducibility* - [GoodScientificCodeWorkshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop)
