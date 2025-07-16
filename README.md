# NeSy-for-graph-data

Code for the paper: *"A Neuro-Symbolic Approach for Probabilistic Reasoning on Graph Data"*

## Repository Structure

### `water-hawqs-github/`
- Contains model definitions, training scripts, and dataset creation tools for the HAWQS dataset.
- Precomputed result graphs are stored as pickle files in **`water-hawqs-github/data`**.
- **Note:** Raw HAWQS/SWAT data is omitted to reduce repository size. The `/scenarios` folder contains only a single example scenario.

### `hetero-hom-experiments/`
- Contains experiments with the Ising model, including dataset generation, model training, and data export for Primula.
- Includes additional experiments using the GMNN model.

### Notebooks
- Both main directories include Python notebooks for data exploration and preprocessing.

### Primula Integration
- Experiments involving Relational Bayesian Networks (RBNs) use [**Primula-3**](https://github.com/manfred-jaeger-aalborg/primula3) for integration.

## Folder Overview

- **water-hawqs-github/**: Main codebase for HAWQS-related experiments and data.
- **water-hawqs-github/data/**: Precomputed results (pickle files) for quick access.
- **water-hawqs-github/scenarios/**: Example scenario for HAWQS (raw data not included).
- **hetero-hom-experiments/**: Scripts and data for Ising model and GMNN experiments.
- **Notebooks/**: Jupyter notebooks for analysis and preprocessing (located in both main folders).

