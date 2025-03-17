# NeSy-for-graph-data
# Code for the Paper: *"A Neuro-Symbolic Approach for Probabilistic Reasoning on Graph Data"*

## water-hawqs-github/
This directory contains the code for parsing, creating, and analyzing data used in the multi-objective experiments. It also includes:

- The `.rbn` (model) file and the `.rdef` (data) file used in Primula.
- Precomputed result graphs, stored as pickle files in **`water-hawqs-github/data`**.  
  *(Note: Raw data generated from HAWQS is not included to save space.)*
- Trained models used for inference, located in **`water-hawqs-github/models`**.
- The actual model implementation, found in **`water-hawqs-github/src/model.py`**.

## hetero-hom-experiments/
This directory contains experiments with different homophily and heterophily settings. It includes:

- The modified implementations of GMNN and GGCN adapted for our experiments.
- Code for the Ising model experiments in **`hetero-hom-experiments/ising`**, with graph generation and training scripts located in **`hetero-hom-experiments/ising/src`**.  
  The trained graphs are later used for inference in Primula.

## Notebooks
Both directories include a Python notebook for a quick overview of the data and some preprocessing.
