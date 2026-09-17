# Fashion Preprocessing

This directory contains the preprocessing, conversion, and validation scripts used to construct the Fashion datasets for the sequential recommendation and knowledge-graph experiments.

The preprocessing pipeline is based on the Amazon `Clothing, Shoes and Jewelry` review and metadata files stored in:

```text
fashion_generalization/data/raw/
```

Three preprocessing configurations are maintained internally. In the paper and
in the reader-facing documentation, the two main Fashion datasets are referred
to as **Fashion-Random** and **Fashion-Heavy**.

The historical internal identifiers are retained in file names, directory names,
and script names for reproducibility:

```text
v1 -> Fashion-Random
v2 -> enriched five-relation extension of Fashion-Random
v3 -> Fashion-Heavy
```
## Data Versions

### Fashion-Random

The standard configuration uses a random sample of up to 50,000 users after iterative 5-core filtering.

The Knowledge Graph contains three relations:

```text
compatible_with
belongs_to
belongs_to_brand
```

The main outputs are stored in:

```text
data/
├── interactions_5_core_filtered.csv
├── metadata_valid_items.json
├── also_buy_pairs.json
│
├── processed/
│   ├── kg_train.tsv
│   ├── taskA_valid.tsv
│   ├── taskA_test.tsv
│   ├── entity2id.tsv
│   ├── relation2id.tsv
│   └── kg_stats.txt
│
└── recbole/
    └── fashion/
        └── fashion.inter
```

### Enriched Fashion-Random Knowledge Graph

This enriched configuration uses the same general preprocessing and user-sampling
strategy as Fashion-Random, but enriches the Knowledge Graph with two additional
relations:

```text
belongs_to_price_tier
belongs_to_pop_tier
```

Price values are discretized into quartiles, while product rank is discretized into popularity tiers.

Intermediate data are stored in:

```text
data_v2/
```

and the processed Knowledge Graph is stored in:

```text
data/processed_v2/
```

The enriched Fashion-Random Knowledge Graph therefore contains five relations:

```text
compatible_with
belongs_to
belongs_to_brand
belongs_to_price_tier
belongs_to_pop_tier
```

### Fashion-Heavy

Fashion-Heavy uses the same five relation types as the enriched Fashion-Random
Knowledge Graph, but changes the user-selection strategy.

Instead of randomly sampling 50,000 users, users are ranked according to:

1. the number of distinct Fashion macro-categories they interacted with;
2. their total number of interactions.

The highest-ranked users are selected, favoring users whose histories provide richer cross-category information for the Knowledge Graph.

After selection, iterative 5-core convergence is checked again.

Intermediate Fashion-Heavy data are stored under the historical path:

```text
data_v3/
```

The processed Knowledge Graph is stored in:

```text
data/processed_v3/
```

and the corresponding RecBole dataset is stored in:

```text
data/recbole_v3/fashion_v3/
```

## KG Construction Scripts

### `build_kg_fashion.py`

Builds the Fashion-Random Knowledge Graph.

Main operations include:

- reading reviews and metadata;
- mapping products to the `top`, `bottom`, and `shoes` macro-categories;
- iterative 5-core filtering;
- random sampling of up to 50,000 users;
- extraction of filtered metadata and `also_buy` relationships;
- construction of the three-relation Knowledge Graph;
- 90/5/5 split of `compatible_with` triples for Task A;
- generation of entity and relation mappings.

Outputs are written to `data/processed/` and intermediate validation data to `data/`.

### `build_kg_fashion_v2.py`

Builds the enriched five-relation Fashion-Random Knowledge Graph.
It follows the v1 pipeline and additionally extracts product price and rank information to construct:

```text
belongs_to_price_tier
belongs_to_pop_tier
```

Outputs are written to `data/processed_v2/`, while intermediate validation files are stored in `data_v2/`.

### `build_kg_fashion_v3.py`

Builds the Fashion-Heavy Knowledge Graph using the category-diversity-based
user-selection strategy.

The graph contains the same five relation types as the enriched Fashion-Random Knowledge Graph, but the user subset is selected according to category diversity and interaction count rather than random sampling.

Outputs are written to `data/processed_v3/`, while intermediate validation files are stored in `data_v3/`.

## RecBole Conversion

### `build_recbole_data.py`

Converts the Fashion-Random filtered interactions into the RecBole atomic `.inter` format.

Input:

```text
data/interactions_5_core_filtered.csv
```

Output:

```text
data/recbole/fashion/fashion.inter
```

The script renames the interaction columns to the RecBole format, removes duplicate user-item interactions, sorts interactions temporally, and reports basic sequence statistics.

### `convert_v3_to_recbole.py`

Converts the Fashion-Heavy interactions into the RecBole format.

Input:

```text
data_v3/interactions_5_core_filtered.csv
```

Output:

```text
data/recbole_v3/fashion_v3/fashion_v3.inter
```

## Knowledge Graph Validation

Two complementary forms of validation are used.

### Semantic KG Validation

`validate_kg_fashion.py`
Validates the Fashion-Random Knowledge Graph against the corresponding intermediate data.

It performs sampled checks on:

- `belongs_to`;
- `belongs_to_brand`;
- `compatible_with`.

It also checks global invariants such as:

- duplicate triples;
- entity-prefix consistency;
- Task A leakage;
- completeness of the `compatible_with` split.

`validate_kg_fashion_v2.py`

Performs the corresponding validation for the enriched Fashion-Random Knowledge Graph and additionally checks:

- `belongs_to_price_tier`;
- `belongs_to_pop_tier`.

### Structural TransE Data Validation

`validate_transe_data.py`

Performs structural checks on the standard v1 TransE input files, including:

- file sizes;
- null values;
- entity mapping coverage;
- relation distribution;
- `compatible_with` node degree;
- train/validation/test leakage;
- validation/test entity coverage;
- cross-category pair distribution.

`validate_transe_data_v2.py`

Runs the corresponding structural checks on the v2 Knowledge Graph stored in `data/processed_v2/`.

## Fashion-Heavy Validation

### `validate_kg_v3.py`

Performs additional checks specifically designed for the Fashion-Heavy configuration.

The checks include:

1. user overlap across v1, v2, and v3;
2. duplicate triples;
3. Task A split leakage;
4. orphan entities in `entity2id.tsv`;
5. item consistency between Task A and Task B;
6. category-diversity distribution in the selected v3 user sample.

### `fix_entity2id_v3.py`

Utility for rebuilding the v3 `entity2id.tsv` using only entities that actually occur in `kg_train.tsv`.

It removes orphan entities from the mapping and creates a backup of the previous mapping before overwriting it.

This script is intended as a repair/cleanup utility and is only required when orphan entities are detected.

## SASRec Data Validation

### `validate_sasrec_data.py`

Validates the standard RecBole dataset:

```text
data/recbole/fashion/fashion.inter
```

The script checks:

- expected RecBole columns;
- null values;
- sequence-length distribution;
- item popularity;
- timestamp range;
- dataset sparsity.

### `validate_inter.py`

Cross-checks the generated `.inter` file against both:

```text
data/interactions_5_core_filtered.csv
```

and the original raw Amazon reviews.

It verifies user and item consistency and performs sequence-level spot checks on randomly sampled users.

## Metadata Inspection

### `inspect_price_rank.py`

Measures the availability of the `price` and `rank` metadata fields in the original Amazon metadata.

This analysis was used to assess whether those attributes had sufficient coverage to justify their inclusion in the enriched v2 and v3 Knowledge Graphs.

## Typical Workflow

### Fashion-Random

```bash
python fashion_generalization/preprocessing/build_kg_fashion.py
python fashion_generalization/preprocessing/validate_kg_fashion.py
python fashion_generalization/preprocessing/validate_transe_data.py
python fashion_generalization/preprocessing/build_recbole_data.py
python fashion_generalization/preprocessing/validate_sasrec_data.py
python fashion_generalization/preprocessing/validate_inter.py
```

### Enriched Fashion-Random KG

```bash
python fashion_generalization/preprocessing/inspect_price_rank.py
python fashion_generalization/preprocessing/build_kg_fashion_v2.py
python fashion_generalization/preprocessing/validate_kg_fashion_v2.py
python fashion_generalization/preprocessing/validate_transe_data_v2.py
```

### Fashion-Heavy

```bash
python fashion_generalization/preprocessing/build_kg_fashion_v3.py
python fashion_generalization/preprocessing/validate_kg_v3.py
python fashion_generalization/preprocessing/convert_v3_to_recbole.py
```

`fix_entity2id_v3.py` should only be used when the v3 validation identifies orphan entities that must be removed from the entity mapping.

## Reproducibility

All preprocessing scripts use repository-relative paths derived from their own file location and therefore do not depend on a user-specific local or HPC directory.

The different data versions are intentionally stored separately:

```text
data/       -> Fashion-Random data and processed outputs
data_v2/    -> intermediate inputs for the enriched Fashion-Random KG
data_v3/    -> intermediate inputs for Fashion-Heavy
```

This separation preserves the exact inputs and outputs associated with each experimental configuration.