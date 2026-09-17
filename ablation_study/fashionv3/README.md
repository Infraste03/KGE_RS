# Fashion-Heavy Knowledge Graph Ablation Study

This directory contains the Knowledge Graph ablation experiments for the
**Fashion-Heavy** dataset.

The goal of these experiments is to measure how individual Knowledge Graph
relations contribute to TransE embedding quality and, subsequently, to the
performance of the joint Knowledge Graph + sequential recommendation model.

Two Knowledge Graph families are considered:

- a **3-relation KG**;
- an enriched **5-relation KG**.

The ablation methodology follows the same general principle used for the B2B
experiments: relation composition is varied while the vocabulary and training
configuration are kept fixed.

Some file names, directory names, and internal experiment identifiers retain
the historical `v3` label used during development to preserve reproducibility
and compatibility with the original experimental pipeline.

## Experimental Methodology

The complete ablation analysis is organized in two stages.

### Stage 1 — Isolated TransE Ablation

TransE is trained independently on different filtered versions of the Fashion
Knowledge Graph.

Only the relation composition changes across variants.

The entity and relation vocabularies are kept fixed across all variants to
ensure consistent embedding indices.

The main link-prediction metrics are:

```text
MRR
MR
Hits@1
Hits@3
Hits@10
```

Both naive and type-constrained evaluations are produced.

For `compatible_with`, the type-constrained evaluation restricts both head and
tail candidates to item entities.

### Stage 2 — Alternate Learning

Selected Knowledge Graph variants can subsequently be used to initialize and
train the joint alternate-learning architecture.

In that setting:

```text
Task A = TransE
Task B = SASRec
```

and the item representations are shared through the common embedding space.

Stage 2 recommendation performance is evaluated using:

```text
Recall@20
NDCG@20
```

Selected Stage 1 variants are propagated to the alternate-learning architecture
to evaluate whether the structural effects observed in isolated TransE training
also influence downstream recommendation performance.

## Directory Structure

```text
fashionv3/
├── README.md
│
├── collect_fashion_ablation.py
├── generate_ablation_plan_v3_3rel.py
├── generate_ablation_plan_v3_5rel.py
├── run_v3_ablation_3rel.py
├── run_v3_ablation_5rel.py
│
├── ablation_3rel/
│   ├── ablation_manifest_3rel.json
│   ├── kg_train_loo_no_belongs_to.tsv
│   ├── kg_train_loo_no_belongs_to_brand.tsv
│   ├── kg_train_loo_no_compatible_with.tsv
│   └── kg_train_targeted_single_cw.tsv
│
├── ablation_5rel/
│   ├── ablation_manifest_5rel.json
│   └── kg_train_<variant>.tsv
│
├── results_3rel/
│   ├── loo_no_belongs_to/
│   ├── loo_no_belongs_to_brand/
│   ├── loo_no_compatible_with/
│   ├── targeted_single_cw/
│   ├── summary.csv
│   └── summary.json
│
└── results_5rel/
    ├── loo_no_belongs_to/
    ├── loo_no_belongs_to_brand/
    ├── loo_no_belongs_to_pop_tier/
    ├── loo_no_belongs_to_price_tier/
    ├── loo_no_compatible_with/
    ├── mix1_keep_cat_pop/
    ├── mix2_keep_pop_cw/
    ├── mix3_keep_cat_brand_cw/
    ├── targeted_pair_cw_brand/
    ├── targeted_pair_cw_category/
    ├── targeted_single_cw/
    ├── targeted_triple_cw_brand_pricetier/
    ├── targeted_triple_cw_category_poptier/
    ├── summary.csv
    └── summary.json
```

## Fashion-Heavy Knowledge Graph

The Fashion-Heavy Knowledge Graph is built from the alternative v3 user-selection
pipeline.

The corresponding processed KG and fixed vocabulary are stored under:

```text
fashion_generalization/data/processed_v3/
```

The ablation scripts use:

```text
entity2id.tsv
relation2id.tsv
taskA_valid.tsv
taskA_test.tsv
```

from this directory.

The entity and relation mappings are not reconstructed from each filtered
ablation graph.

This is important because all variants must preserve the same embedding
indices and dimensional structure.

## 3-Relation Knowledge Graph

The 3-relation configuration contains:

```text
compatible_with
belongs_to
belongs_to_brand
```

where:

- `compatible_with` represents product compatibility derived from item
  relationships;
- `belongs_to` connects items to product categories;
- `belongs_to_brand` connects items to brands.

### 3-Relation Ablation Variants

Four variants are generated.

#### Leave-One-Out

```text
loo_no_compatible_with
loo_no_belongs_to
loo_no_belongs_to_brand
```

Each variant removes exactly one relation from the full 3-relation graph.

#### Targeted Variant

```text
targeted_single_cw
```

This configuration retains only:

```text
compatible_with
```

It is used to test whether the compatibility relation alone can preserve a
large portion of the useful KG signal.

### 3-Relation Manifest

The generated configurations are described in:

```text
ablation_3rel/ablation_manifest_3rel.json
```

For each variant, the manifest records:

```text
kg_train_ablation
relations_kept
relations_excluded
kind
```

## 5-Relation Knowledge Graph

The enriched 5-relation configuration contains:

```text
compatible_with
belongs_to
belongs_to_brand
belongs_to_price_tier
belongs_to_pop_tier
```

The two additional relations introduce engineered product attributes:

- `belongs_to_price_tier`: item to price-tier relationship;
- `belongs_to_pop_tier`: item to popularity-tier relationship.

The purpose of this configuration is to evaluate whether structured price and
popularity information provides useful complementary signal beyond
compatibility, category, and brand.

## 5-Relation Ablation Variants

The default 5-relation ablation plan contains **13 variants**:

```text
5 leave-one-out
3 random-mix
5 targeted
```

### Leave-One-Out Variants

```text
loo_no_compatible_with
loo_no_belongs_to
loo_no_belongs_to_brand
loo_no_belongs_to_price_tier
loo_no_belongs_to_pop_tier
```

Each configuration removes one relation while retaining the other four.

### Random-Mix Variants

The current generated random-mix configurations are:

```text
mix1_keep_cat_pop
mix2_keep_pop_cw
mix3_keep_cat_brand_cw
```

These configurations retain randomly selected subsets of the five available
relations.

The random combinations are generated deterministically when the same random
seed is used.

### Targeted Variants

The targeted configurations are centered on `compatible_with`:

```text
targeted_pair_cw_brand
targeted_pair_cw_category
targeted_single_cw
targeted_triple_cw_brand_pricetier
targeted_triple_cw_category_poptier
```

These variants are designed to evaluate which types of metadata provide the
most useful complementary information when combined with product
compatibility.

## Ablation Plan Generation

### 3-Relation Plan

The script:

```text
generate_ablation_plan_v3_3rel.py
```

generates the four 3-relation variants and writes:

```text
ablation_3rel/ablation_manifest_3rel.json
ablation_3rel/kg_train_<variant>.tsv
```

Example execution from `ablation_study/fashionv3/`:

```bash
python generate_ablation_plan_v3_3rel.py \
    --kg-train ../../fashion_generalization/data/processed_v3/kg_train.tsv \
    --output-dir ablation_3rel \
    --manifest ablation_3rel/ablation_manifest_3rel.json
```

### 5-Relation Plan

The script:

```text
generate_ablation_plan_v3_5rel.py
```

generates the 5 leave-one-out, 3 random-mix, and 5 targeted configurations.

Example execution:

```bash
python generate_ablation_plan_v3_5rel.py \
    --kg-train ../../fashion_generalization/data/processed_v3/kg_train.tsv \
    --output-dir ablation_5rel \
    --manifest ablation_5rel/ablation_manifest_5rel.json \
    --seed 42
```

## TransE Training

### 3-Relation Experiments

The script:

```text
run_v3_ablation_3rel.py
```

trains TransE independently on each 3-relation ablation variant.

The fixed TransE configuration is:

```text
embedding_dim     = 128
learning_rate     = 0.000688578505982584
num_negs_per_pos  = 8
loss              = marginranking
loss_margin       = 8.95047357618049
batch_size        = 1024
max epochs        = 300
random seed       = 42
```

Early stopping is enabled.

The hyperparameters are fixed from the TransE HPO performed on the
corresponding Fashion-Heavy 3-relation family.

### 5-Relation Experiments

The script:

```text
run_v3_ablation_5rel.py
```

uses:

```text
embedding_dim     = 128
learning_rate     = 0.0003048761416961341
num_negs_per_pos  = 8
loss              = marginranking
loss_margin       = 9.671773805079942
batch_size        = 1024
max epochs        = 300
random seed       = 42
```

Again, the configuration is fixed across all ablation variants.

No new HPO is performed for individual variants.

## Fixed Vocabulary

A central requirement of the experiment is that every ablation variant uses
the same:

```text
entity_to_id
relation_to_id
```

mapping.

The vocabulary is loaded from:

```text
fashion_generalization/data/processed_v3/entity2id.tsv
fashion_generalization/data/processed_v3/relation2id.tsv
```

rather than inferred independently from each filtered KG.

This ensures that:

```text
num_entities
num_relations
entity indices
relation indices
```

remain consistent across all variants.

This alignment is particularly important when the resulting TransE
representations are used as initialization for the shared embedding layer in
the alternate-learning architecture.

## Output of Each TransE Variant

Each experiment produces a dedicated result directory.

For example:

```text
results_5rel/loo_no_belongs_to/
```

contains:

```text
best_metrics_loo_no_belongs_to.json
best_model_loo_no_belongs_to.pt
run_config_loo_no_belongs_to.json
run_loo_no_belongs_to.log
```

The same naming convention is used for all variants.

### `best_model_<variant>.pt`

Contains the trained TransE model state dictionary.

### `best_metrics_<variant>.json`

Contains the final test-set evaluation metrics, including:

```text
naive
type_constrained
```

evaluation results.

### `run_config_<variant>.json`

Records the configuration required to identify and reproduce the run,
including:

```text
variant
kind
relations_kept
relations_excluded
TransE hyperparameters
random seed
training epochs
batch size
```

### `run_<variant>.log`

Contains the execution log for the corresponding experiment.

## Result Aggregation

The script:

```text
collect_fashion_ablation.py
```

collects the `best_metrics_<variant>.json` files from:

```text
results_3rel/
results_5rel/
```

and extracts the type-constrained:

```text
MRR
Hits@10
```

metrics.

The individual training scripts also generate:

```text
summary.csv
summary.json
```

inside the corresponding result directories.

## Results

### Stage 1 — Isolated TransE: 3-Relation KG

The following table reports the type-constrained link-prediction results for
the full 3-relation Knowledge Graph and its ablation variants.

| KG configuration | MRR | Hits@10 |
|---|---:|---:|
| Full 3-rel graph | 0.1010 | 0.1840 |
| Without `compatible_with` | **0.0004** | **0.0006** |
| Without `belongs_to` | 0.0874 | 0.1707 |
| Without `belongs_to_brand` | 0.0844 | 0.1562 |
| `compatible_with` only | 0.0654 | 0.1250 |

The strongest degradation occurs when `compatible_with` is removed:

```text
Full 3-rel KG:
MRR     = 0.1010
Hits@10 = 0.1840

Without compatible_with:
MRR     = 0.0004
Hits@10 = 0.0006
```

This result identifies compatibility as the dominant relation for the isolated
Fashion-Heavy link-prediction task.

Retaining only `compatible_with` performs substantially better than removing it
completely, but it does not recover the performance of the full Knowledge Graph.

### Stage 1 — Isolated TransE: 5-Relation KG

| KG configuration | MRR | Hits@10 |
|---|---:|---:|
| Full 5-rel graph | 0.1033 | 0.1852 |
| Without `compatible_with` | 0.0005 | 0.0006 |
| Without `belongs_to` | 0.0938 | 0.1672 |
| Without `belongs_to_brand` | 0.0875 | 0.1453 |
| Without `belongs_to_price_tier` | 0.1041 | 0.1788 |
| Without `belongs_to_pop_tier` | 0.1050 | 0.1881 |
| `belongs_to` + `belongs_to_pop_tier` | 0.0005 | 0.0012 |
| `belongs_to_pop_tier` + `compatible_with` | 0.0757 | 0.1337 |
| `belongs_to` + `belongs_to_brand` + `compatible_with` | **0.1122** | **0.1875** |
| `compatible_with` + `belongs_to_brand` | 0.0989 | 0.1771 |
| `compatible_with` + `belongs_to` | 0.0902 | 0.1516 |
| `compatible_with` only | 0.0688 | 0.1215 |
| `compatible_with` + `belongs_to_brand` + `belongs_to_price_tier` | 0.0926 | 0.1632 |
| `compatible_with` + `belongs_to` + `belongs_to_pop_tier` | 0.0872 | 0.1499 |

The removal of `compatible_with` again produces an almost complete collapse in
link-prediction performance.

Interestingly, removing the price-tier and popularity-tier relations increases
the isolated TransE performance from MRR 0.1033 and Hits@10 0.1852 for the full
five-relation graph to MRR 0.1122 and Hits@10 0.1875.


This indicates that adding more relations does not automatically improve
Knowledge Graph embedding quality.

The price-tier and popularity-tier relations are not individually essential:
removing either relation leaves isolated TransE performance close to, or
slightly above, the full 5-relation configuration.

### Stage 2 — Alternate Learning

Selected Stage 1 variants were propagated to the downstream recommendation
architecture.

These ablation experiments use:

```text
seed       = 2020
scheduling = one_to_one
```

and are therefore single-seed diagnostic analyses.

| KG configuration | KG | Recall@20 | NDCG@20 |
|---|---|---:|---:|
| Full graph | 5-rel | 0.09786 | 0.08039 |
| Without `compatible_with` | 5-rel | 0.09384 | 0.07900 |
| Without price-tier and popularity-tier | 5-rel | 0.09766 | 0.08033 |
| `compatible_with` + `belongs_to_brand` | 5-rel | 0.09728 | 0.08024 |
| Full graph | 3-rel | **0.09822** | **0.08060** |
| Without `belongs_to` | 3-rel | 0.09692 | 0.08009 |


Removing `compatible_with` from the five-relation graph reduces Recall@20 from
0.09786 to 0.09384 and NDCG@20 from 0.08039 to 0.07900, corresponding to
relative reductions of approximately 4.1% and 1.7%, respectively.

These downstream ablations use a single seed and should therefore be interpreted
as diagnostic comparisons. Nevertheless, the result is consistent with the B2B
findings and further highlights the importance of compatibility information
across both domains.

In isolated TransE evaluation, removing the price-tier and popularity-tier
relations improves MRR from 0.1033 to 0.1122. However, the same reduced graph
achieves Recall@20 of 0.09766 and NDCG@20 of 0.08033 downstream, compared with
0.09786 and 0.08039 for the complete five-relation graph.

Therefore, the improvement observed in isolated Knowledge Graph embedding does
not transfer to downstream recommendation.

This demonstrates that:

```text
better isolated KGE performance
does not necessarily imply
better recommendation performance
```

The full 3-relation and 5-relation models are also very close in downstream
performance.

The corresponding five-seed full-model results are:

| KG Family | Recall@20 | NDCG@20 |
|---|---:|---:|
| 3-rel | **0.09796 ± 0.00028** | **0.08046 ± 0.00014** |
| 5-rel | 0.09783 ± 0.00035 | 0.08041 ± 0.00008 |

Their difference is not statistically significant:

```text
Recall@20: p = 0.8125
NDCG@20:   p = 0.6250
```

This suggests that the engineered price-tier and popularity-tier relations do
not provide a measurable downstream recommendation advantage over the simpler
3-relation Knowledge Graph.

## Reproducibility

The Python scripts derive repository paths from their own location rather than
using user-specific absolute paths.

For example, the repository root is resolved programmatically from:

```python
Path(__file__).resolve()
```

This keeps the source code portable across local machines and HPC systems.

Cluster-specific execution details, when required, should be handled by the
corresponding SLURM scripts rather than hard-coded into the Python source.

## Relationship with the Main Fashion Experiments

The experiments in this directory correspond specifically to the
**Fashion-Heavy Knowledge Graph ablation study**.

The main Fashion implementation, preprocessing pipeline, pretrained models,
and alternate-learning architecture are documented under:

```text
fashion_generalization/
```

This directory isolates the KG relation-composition analysis so that the
effect of each structural component can be studied independently from the
rest of the Fashion experimental pipeline.