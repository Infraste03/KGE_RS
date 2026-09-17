# Knowledge Graph Ablation Study

This directory contains the ablation experiments used to analyze the contribution
of individual Knowledge Graph relations to both:

1. Knowledge Graph embedding quality;
2. downstream recommendation performance.

The analysis covers three experimental settings:

```text
B2B
Fashion-Random
Fashion-Heavy
```

In the paper, the two Fashion settings are referred to as **Fashion-Random**
and **Fashion-Heavy**. Some file names, directory names, and internal experiment
identifiers retain the historical labels used during development (e.g., `v1`
and `v3`) to preserve reproducibility and compatibility with the original
experimental pipeline.

The general methodology is organized into two stages.

## Experimental Methodology

### Stage 1 — Isolated TransE Ablation

In the first stage, TransE is trained independently on different filtered
versions of the Knowledge Graph.

The purpose is to evaluate how the removal or retention of specific relations
affects Knowledge Graph embedding quality.

The main evaluation metrics are:

```text
MRR
MR
Hits@1
Hits@3
Hits@10
```

Both naive and type-constrained evaluations are used where applicable.

The main comparison reported in this repository focuses on:

```text
MRR
Hits@10
```

### Stage 2 — Alternate-Learning Ablation

In the second stage, selected Knowledge Graph variants are integrated into the
joint alternate-learning recommendation architecture.

The architecture combines:

```text
Task A = TransE
Task B = SASRec
```

through a shared embedding space.

This stage measures whether the structural information identified as useful in
the isolated KG experiments also affects recommendation quality.

The primary recommendation metrics are:

```text
Recall@20
NDCG@20
```

When multiple random seeds are available, results are reported as:

```text
mean ± standard deviation
```

## Directory Structure

```text
ablation_study/
├── README.md
│
├── generate_ablation_plan.py
├── generate_ablation_plan_extra.py
├── run_transe_ablation_batch.py
├── reproduce_baseline_transe.py
├── collect_results.py
├── ablation_summary.csv
│
├── SBATCH FILE/
│   ├── README.md
│   ├── sbatch_transe_ablation.sh
│   ├── sbatch_v3_ablation_3rel.sh
│   └── sbatch_v3_ablation_5rel.sh
│
├── fashionv3/
│   ├── README.md
│   ├── generate_ablation_plan_v3_3rel.py
│   ├── generate_ablation_plan_v3_5rel.py
│   ├── run_v3_ablation_3rel.py
│   ├── run_v3_ablation_5rel.py
│   ├── collect_fashion_ablation.py
│   ├── ablation_3rel/
│   ├── ablation_5rel/
│   ├── results_3rel/
│   └── results_5rel/
│
└── step4_alternate_results/
    ├── README.md
    ├── b2b/
    └── fashion_v1/
```

The filtered B2B Knowledge Graph variants used by the ablation scripts are
generated under the repository-level processed-data directory:

```text
data/processed/ablation/
```

The isolated TransE B2B outputs are stored separately under the repository-level
results directory.

## B2B Knowledge Graph

The B2B Knowledge Graph contains five relations:

```text
bought
compatible_with
compatible_with_model
instance_of
owns
```

The ablation analysis investigates which of these relations provide the most
useful structural information.

## B2B Ablation Plan

The B2B Stage 1 ablation plan contains:

```text
5 leave-one-out variants
3 random-mix variants
5 targeted variants
```

for a total of:

```text
13 ablation variants
```

in addition to the full Knowledge Graph reference configuration.

### Leave-One-Out Variants

```text
loo_no_bought
loo_no_compatible_with
loo_no_compatible_with_model
loo_no_instance_of
loo_no_owns
```

Each variant removes exactly one relation from the full graph.

### Random-Mix Variants

```text
mix1_keep_compatible_with_owns
mix2_keep_bought_owns
mix3_keep_bought_compatible_with
```

These experiments retain different subsets of the available relations.

### Targeted Variants

The targeted configurations are centered on `compatible_with`:

```text
targeted_pair_cw_cwm
targeted_pair_cw_instance
targeted_single_cw
targeted_triple_cw_cwm_owns
targeted_triple_cw_bought_instance
```

These configurations were introduced to analyze whether compatibility
information is sufficient by itself or becomes more useful when combined with
specific complementary relations.

## B2B Stage 1 — Isolated TransE Results

The following table reports the TransE results obtained from the full Knowledge
Graph and the 13 ablation variants.

The reported values correspond to the type-constrained evaluation.

| Relations retained | Type | MRR | Hits@10 |
|---|---|---:|---:|
| All five relations | full graph | 0.2953 | 0.5020 |
| `compatible_with` + `compatible_with_model` + `instance_of` + `owns` | leave-one-out | 0.2285 | 0.4083 |
| `bought` + `compatible_with_model` + `instance_of` + `owns` | leave-one-out | 0.0151 | 0.0321 |
| `bought` + `compatible_with` + `instance_of` + `owns` | leave-one-out | 0.2962 | 0.4999 |
| `bought` + `compatible_with` + `compatible_with_model` + `owns` | leave-one-out | 0.2988 | 0.5082 |
| `bought` + `compatible_with` + `compatible_with_model` + `instance_of` | leave-one-out | 0.2931 | 0.5078 |
| `compatible_with` + `owns` | reduced graph | 0.2139 | 0.3778 |
| `bought` + `owns` | reduced graph | 0.0111 | 0.0229 |
| `bought` + `compatible_with` | reduced graph | 0.2938 | 0.5026 |
| `compatible_with` + `compatible_with_model` | targeted graph | 0.2224 | 0.3978 |
| `compatible_with` + `instance_of` | targeted graph | 0.2051 | 0.3629 |
| `compatible_with` only | targeted graph | 0.2137 | 0.3837 |
| `compatible_with` + `bought` + `instance_of` | targeted graph | 0.2939 | 0.5005 |
| `compatible_with` + `compatible_with_model` + `owns` | targeted graph | 0.2254 | 0.3999 |

The strongest degradation occurs when `compatible_with` is removed.

In particular:

```text
full KG
MRR     = 0.2953
Hits@10 = 0.5020

without compatible_with
MRR     = 0.0151
Hits@10 = 0.0321
```

This indicates that `compatible_with` is the dominant relation for the
link-prediction task considered in the B2B Knowledge Graph.

The reduced graph retaining only `bought` and `owns` also produces a strong
collapse, showing that retaining fewer relations is not sufficient unless the
relevant structural information is preserved.

Conversely, removing `instance_of` or `owns`, retaining only `bought` and
`compatible_with`, or retaining `compatible_with`, `bought`, and `instance_of`
preserves performance close to the complete graph.

## B2B Stage 2 — Alternate-Learning Results

Based on the Stage 1 analysis, a subset of B2B variants was propagated to the
joint alternate-learning model.

The selected configurations were:

```text
loo_no_compatible_with
loo_no_instance_of
mix1_keep_compatible_with_owns
mix3_keep_bought_compatible_with
targeted_triple_cw_bought_instance
```

All reported runs use the `one_to_one` scheduling strategy.

The full Knowledge Graph model is used as the reference.

| Relations retained | Recall@20 | NDCG@20 |
|---|---:|---:|
| All five relations | 0.4612 ± 0.0177 | 0.2773 ± 0.0098 |
| `bought` + `owns` + `instance_of` + `compatible_with_model` | 0.4247 ± 0.0173 | 0.2545 ± 0.0141 |
| `bought` + `owns` + `compatible_with` + `compatible_with_model` | 0.4658 ± 0.0126 | 0.2774 ± 0.0072 |
| `compatible_with` + `owns` | 0.4338 ± 0.0187 | 0.2665 ± 0.0072 |
| `compatible_with` + `bought` + `instance_of` | 0.4566 ± 0.0150 | 0.2691 ± 0.0096 |
| `bought` + `compatible_with` | 0.4612 ± 0.0139 | 0.2771 ± 0.0122 |

The Stage 2 results further highlight the importance of relation composition.

Removing `compatible_with` produces the clearest reduction in recommendation
performance, consistent with the isolated TransE results.

At the same time, removing `instance_of` does not negatively affect the
recommendation model and results in performance slightly above the complete
Knowledge Graph configuration.

The configuration retaining:

```text
bought
compatible_with
```

also reaches recommendation performance very close to the full graph.

Overall, these experiments indicate that the usefulness of the Knowledge Graph
depends primarily on **which relations are available**, rather than simply on
the number of relations included.

## B2B Baseline Reproduction

The script:

```text
reproduce_baseline_transe.py
```

retrains TransE on the complete 5-relation B2B Knowledge Graph using the
previously selected HPO configuration.

It is intended as a sanity check before running the ablation experiments.

The fixed parameters are:

```text
embedding_dim     = 400
learning_rate     = 0.0008798929749689024
num_negs_per_pos  = 128
loss              = marginranking
loss_margin       = 5.191058165461712
random_seed       = 42
```

The expected type-constrained reference performance is approximately:

```text
MRR     = 0.2953
Hits@10 = 0.5020
```

## B2B Ablation Generation

### Initial Ablation Plan

The script:

```text
generate_ablation_plan.py
```

generates:

```text
5 leave-one-out variants
3 random-mix variants
```

and produces the corresponding filtered KG training files.

Example execution from the repository root:

```bash
python ablation_study/generate_ablation_plan.py \
    --kg-train data/processed/kg_train.tsv \
    --output-dir data/processed/ablation \
    --manifest data/processed/ablation/ablation_manifest.json \
    --seed 42
```

### Additional Targeted Variants

The script:

```text
generate_ablation_plan_extra.py
```

extends the existing manifest with the five targeted configurations centered
on `compatible_with`.

Example:

```bash
python ablation_study/generate_ablation_plan_extra.py \
    --kg-train data/processed/kg_train.tsv \
    --output-dir data/processed/ablation \
    --manifest data/processed/ablation/ablation_manifest.json
```

Existing variants are preserved and are not overwritten.

## B2B TransE Ablation Training

The script:

```text
run_transe_ablation_batch.py
```

trains TransE on the generated B2B variants.

A key requirement is that the entity and relation vocabularies remain fixed
across all ablation configurations.

The vocabulary is therefore reconstructed from the original complete Knowledge
Graph and reused for every filtered variant.

This ensures consistent:

```text
entity indices
relation indices
num_entities
num_relations
embedding dimensions
```

which is required when the pretrained TransE representations are later used in
the Step 4 shared embedding space.

The fixed training parameters are:

```text
embedding_dim     = 400
learning_rate     = 0.0008798929749689024
num_negs_per_pos  = 128
loss              = marginranking
loss_margin       = 5.191058165461712
batch_size        = 1024
max epochs        = 300
random seed       = 42
```

Early stopping is enabled.

Each variant produces files such as:

```text
best_model_<variant>.pt
best_metrics_<variant>.json
run_config_<variant>.json
context_<variant>.json
run_<variant>.log
```

Summary files are also generated for the executed variants.

## Fashion-Random Ablation

Fashion-Random uses the randomly sampled user subset described in the paper.
The isolated Stage 1 analysis reported here uses the standard three-relation
Fashion Knowledge Graph.

The historical directory:

```text
step4_alternate_results/fashion_v1/
```

also contains single-seed downstream diagnostics for an enriched five-relation
extension of the same Random user sample. The directory name retains the
historical `fashion_v1` identifier for reproducibility.

### Fashion-Random Stage 1 — Isolated TransE

The isolated TransE ablation evaluates the effect of removing individual
structural relations from the standard Fashion Knowledge Graph.

| KG configuration | MRR |
|---|---:|
| Full graph | 0.1057 |
| Without `belongs_to_brand` | 0.0907 |
| Without `belongs_to` | 0.0946 |
| Without `compatible_with` | **0.0023** |

The largest degradation occurs when `compatible_with` is removed.

This relation is derived from cross-category `also_buy` information and
provides the main item-to-item structural signal for the Fashion link-prediction
task.

### Fashion-Random Auxiliary Stage 2 Diagnostics

The historical enriched five-relation downstream ablations were evaluated with
both `epoch` and `one_to_one` scheduling and use Cross-Entropy for Task B.
These experiments use a single random seed and should therefore be interpreted
as diagnostic comparisons rather than multi-seed estimates.

| KG configuration | Scheduling | Recall@20 | NDCG@20 |
|---|---|---:|---:|
| Full 5-rel graph | `epoch` | 0.1235 | **0.1100** |
| Full 5-rel graph | `one_to_one` | 0.1235 | 0.1099 |
| Without `belongs_to_brand` | `epoch` | 0.1162 | 0.1063 |
| Without `belongs_to_brand` | `one_to_one` | 0.1165 | 0.1065 |
| Without `belongs_to` | `epoch` | 0.1235 | 0.1075 |
| Without `belongs_to` | `one_to_one` | 0.1234 | 0.1079 |
| Without `compatible_with` | `epoch` | 0.1209 | 0.1083 |
| Without `compatible_with` | `one_to_one` | 0.1209 | 0.1086 |

These auxiliary results further illustrate that isolated Knowledge Graph
embedding quality and downstream recommendation quality need not change by the
same amount.

## Fashion-Heavy Ablation

Fashion-Heavy contains a dedicated ablation pipeline under the historical path:

```text
ablation_study/fashionv3/
```

Two Knowledge Graph families are analyzed:

```text
3-relation KG
5-relation KG
```

The 3-relation family contains:

```text
compatible_with
belongs_to
belongs_to_brand
```

The 5-relation family additionally introduces:

```text
belongs_to_price_tier
belongs_to_pop_tier
```

The complete Fashion-Heavy methodology and directory organization are documented
in:

```text
fashionv3/README.md
```

### Fashion-Heavy Stage 1 — 3-Relation TransE

| Relations retained | Type | MRR | Hits@10 |
|---|---|---:|---:|
| All three relations | full graph | 0.1010 | 0.1840 |
| `belongs_to` + `belongs_to_brand` | leave-one-out | **0.0004** | **0.0006** |
| `compatible_with` + `belongs_to_brand` | leave-one-out | 0.0874 | 0.1707 |
| `compatible_with` + `belongs_to` | leave-one-out | 0.0844 | 0.1562 |
| `compatible_with` only | targeted graph | 0.0654 | 0.1250 |

Removing `compatible_with` produces an almost complete collapse in isolated
link-prediction performance.

The result confirms that compatibility is the dominant structural relation
within the 3-relation Fashion-Heavy Knowledge Graph.

### Fashion-Heavy Stage 1 — 5-Relation TransE

| Relations retained | Type | MRR | Hits@10 |
|---|---|---:|---:|
| All five relations | full graph | 0.1033 | 0.1852 |
| `belongs_to` + `belongs_to_brand` + `belongs_to_price_tier` + `belongs_to_pop_tier` | leave-one-out | 0.0005 | 0.0006 |
| `compatible_with` + `belongs_to_brand` + `belongs_to_price_tier` + `belongs_to_pop_tier` | leave-one-out | 0.0938 | 0.1672 |
| `compatible_with` + `belongs_to` + `belongs_to_price_tier` + `belongs_to_pop_tier` | leave-one-out | 0.0875 | 0.1453 |
| `compatible_with` + `belongs_to` + `belongs_to_brand` + `belongs_to_pop_tier` | leave-one-out | 0.1041 | 0.1788 |
| `compatible_with` + `belongs_to` + `belongs_to_brand` + `belongs_to_price_tier` | leave-one-out | 0.1050 | 0.1881 |
| `belongs_to` + `belongs_to_pop_tier` | reduced graph | 0.0005 | 0.0012 |
| `belongs_to_pop_tier` + `compatible_with` | reduced graph | 0.0757 | 0.1337 |
| `belongs_to` + `belongs_to_brand` + `compatible_with` | reduced graph | **0.1122** | **0.1875** |
| `compatible_with` + `belongs_to_brand` | targeted graph | 0.0989 | 0.1771 |
| `compatible_with` + `belongs_to` | targeted graph | 0.0902 | 0.1516 |
| `compatible_with` only | targeted graph | 0.0688 | 0.1215 |
| `compatible_with` + `belongs_to_brand` + `belongs_to_price_tier` | targeted graph | 0.0926 | 0.1632 |
| `compatible_with` + `belongs_to` + `belongs_to_pop_tier` | targeted graph | 0.0872 | 0.1499 |

Removing the price-tier and popularity-tier relations while retaining
`compatible_with`, `belongs_to`, and `belongs_to_brand` achieves the strongest
isolated TransE result:

```text
MRR     = 0.1122
Hits@10 = 0.1875
```

compared with:

```text
Full 5-rel KG
MRR     = 0.1033
Hits@10 = 0.1852
```

Therefore, the complete 5-relation graph is not necessarily the optimal graph
for isolated link prediction.

At the same time, the collapse observed when `compatible_with` is removed
confirms the central role of compatibility information.

### Fashion-Heavy Stage 2 — Alternate Learning

Selected Stage 1 variants were propagated to the downstream alternate-learning
architecture.

These experiments use:

```text
seed       = 2020
scheduling = one_to_one
```

and are therefore single-seed ablation comparisons.

| KG family | Configuration | Recall@20 | NDCG@20 |
|---|---|---:|---:|
| 5-rel | Full graph | 0.09786 | 0.08039 |
| 5-rel | Without `compatible_with` | 0.09384 | 0.07900 |
| 5-rel | Without price-tier and popularity-tier | 0.09766 | 0.08033 |
| 5-rel | `compatible_with` + `belongs_to_brand` | 0.09728 | 0.08024 |
| 3-rel | Full graph | **0.09822** | **0.08060** |
| 3-rel | Without `belongs_to` | 0.09692 | 0.08009 |

Removing the price-tier and popularity-tier relations improves isolated
TransE MRR from 0.1033 to 0.1122, but this advantage does not transfer
downstream: Recall@20 changes from 0.09786 to 0.09766 and NDCG@20 from
0.08039 to 0.08033.

This shows that improved isolated Knowledge Graph embedding quality does not
necessarily imply improved downstream sequential recommendation performance.

By contrast, removing `compatible_with` from the five-relation graph reduces
Recall@20 from 0.09786 to 0.09384 and NDCG@20 from 0.08039 to 0.07900,
corresponding to relative reductions of approximately 4.1% and 1.7%,
respectively. These downstream ablations use a single seed and should therefore
be interpreted as diagnostic comparisons. Nevertheless, the result is
consistent with the B2B findings and further highlights the importance of
compatibility information across both domains.

The downstream results also show that the full three-relation and five-relation
configurations perform very similarly, suggesting that the engineered
price-tier and popularity-tier relations provide limited additional benefit to
the alternate-learning recommender.


## Result Collection

The script:

```text
collect_results.py
```

scans:

```text
step4_alternate_results/
```

for `best_metrics.json` files and generates:

```text
ablation_summary.csv
```

It currently recognizes result-directory patterns for:

```text
b2b
fashion_v1
```

and extracts known recommendation metrics such as:

```text
Recall@20
NDCG@20
Recall@10
NDCG@10
MRR
```

The Fashion-Heavy TransE results use their dedicated collection script,
retained under the historical `fashionv3/` path:

```text
fashionv3/collect_fashion_ablation.py
```

## SLURM Execution

The directory:

```text
SBATCH FILE/
```

contains the job-array scripts used for the HPC executions.

The public scripts do not contain user-specific paths.

Instead, they use the placeholder:

```text
/path/to/KG_RS
```

which must be replaced with the absolute repository path on the target HPC
system before execution.

Further information is available in:

```text
SBATCH FILE/README.md
```

## Relationship with the Main Experimental Pipelines

This directory focuses specifically on the **Knowledge Graph relation ablation
analysis**.

The complete implementations of the recommendation architectures are maintained
in:

```text
B2B_alternate_learning/
fashion_generalization/
```

The main processed B2B data are stored under:

```text
data/
```

while the Fashion preprocessing, model training, and alternate-learning
experiments are documented under:

```text
fashion_generalization/
```

This separation keeps the ablation study reproducible while avoiding duplication
of the main training pipelines.
