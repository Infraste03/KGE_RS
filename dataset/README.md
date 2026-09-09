# Dataset and Preprocessing Configurations

This directory contains the dataset splits and preprocessed files utilized for the empirical evaluation presented in the study. All data has been rigorously processed and formatted to comply with the input requirements of the benchmarked recommendation models.

## Data Ecosystem

To ensure reproducibility and standard evaluation protocols, the interaction data has been transformed into the atomic format required by the **RecBole** framework, which serves as the core library for our baseline algorithms.

The directory is structured as follows:

### 1. Raw Splits and Base Files
* `full_dataset.csv`: The complete, aggregated interaction dataset after the initial data cleaning and filtering procedures.
* `train.csv`, `validation.csv`, `test.csv`: The foundational splits utilized to evaluate the models' generalization capabilities. These splits ensure that all models are benchmarked against the exact same interaction contexts to prevent data leakage and guarantee a fair comparison.

### 2. RecBole-Compatible Formats
Model-specific folders containing `.inter`, `.item`, and feature files seamlessly readable by the RecBole data loaders:
* **`b2b_data/`**: Contains the standard sequential and implicit feedback representations used by the majority of the benchmark models (e.g., SASRec, BERT4Rec, GRU4Rec, NARM, NextItNet, FPMC).
* **`b2b_data_temporal/`**: Retains rigid temporal dynamics and timestamps, formatted for models that strongly rely on absolute time intervals rather than just positional sequences.
* **`b2b_data_fdsa/`**: Contains item-feature augmented data formatted specifically for the FDSA (Feature-level Deeper Self-Attention) model, incorporating additional metadata attributes essential for feature-aware sequential recommendations.
