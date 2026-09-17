# Fashion Alternate Learning

This directory contains the implementation of the alternate-learning
experiments for the Fashion generalization study.

The architecture combines:

- **Task A:** Knowledge Graph Embedding with TransE
- **Task B:** sequential recommendation with SASRec
- **SharedEmbedding:** a common item/entity embedding space jointly updated
  by both tasks

The repository contains experiments for the two Fashion settings considered in
the paper: **Fashion-Random** and **Fashion-Heavy**.

Some file names, directory names, and internal experiment identifiers retain
the historical `v3` label used during development to preserve reproducibility
and compatibility with the original experimental pipeline.

## Directory Structure

```text
alternate_learning/
├── checks/
├── configs/
├── hpc/
├── models/
├── variants/
│   ├── ablation_3rel/
│   ├── five_rel/
│   └── v3/
├── sasrec_fashion_seeds_results/
├── hpc_step4_results/
├── hpc_step4_results_v3/
├── hpo_results_fashion/
├── hpo_results_fashion_ce/
├── data_loaders_fashion.py
├── eval_utils_fashion.py
├── load_kg_fashion.py
├── unified_id_space_fashion.py
├── alternate_loop_fashion.py
├── alternate_loop_fashion_ce.py
├── run_step4_fashion.py
├── run_step4_fashion_ce.py
├── run_hpo_step4_fashion.py
└── run_hpo_step4_fashion_ce.py
```

Generated result directories may not be included in the final public
repository and can instead be archived separately.

## Fashion-Random

Fashion-Random represents the original generalization experiment.

Two Task B training objectives were explored:

- BPR loss
- Cross-Entropy loss

The main runners are:

```text
run_step4_fashion.py
run_step4_fashion_ce.py
```

with the corresponding HPO wrappers:

```text
run_hpo_step4_fashion.py
run_hpo_step4_fashion_ce.py
```

Pretrained TransE and SASRec checkpoints are specified through the YAML
configuration files under `configs/`.

The canonical Fashion-Random pretrained checkpoints are:

```text
results/hpo_transe/best_model.pt
results/sasrec_hpo/trial_017/best_valid.pth
```

## Fashion-Heavy

Fashion-Heavy extends the original setup with an updated Knowledge Graph and
additional relation/entity information.

The corresponding implementation is located under:

```text
variants/v3/
```

Fashion-Heavy supports:

- 3-relation KG experiments
- 5-relation KG experiments
- relation-ablation experiments
- dedicated HPO
- a dedicated unified ID space and evaluation pipeline

See `variants/v3/README.md` for additional details.

## Other Variants

`variants/five_rel/` contains the earlier five-relation Fashion extension.

`variants/ablation_3rel/` contains the Fashion-Random three-relation
leave-one-relation-out ablation experiments.

## Evaluation

Task B is evaluated using:

- Recall@20
- NDCG@20

Model selection is based on validation performance.

The final test metrics are computed from the validation-selected model.

## Experimental Results

Final multi-seed experiments use the following five seeds:

```text
2020
42
1024
999
2024
```

Multi-seed results are reported as mean ± standard deviation.

### Fashion-Random

Fashion-Random compares BPR and Cross-Entropy as Task B objectives.

The initial HPO results are:

| Model | Scheduling | Task B Loss | Recall@20 | NDCG@20 |
|---|---|---|---:|---:|
| SASRec baseline | - | CE | 0.0966 ± 0.00023 | 0.0889 ± 0.00016 |
| KGSEQ | `one_to_one` | BPR | 0.0946 | 0.0501 |
| KGSEQ | `epoch` | BPR | 0.1017 | 0.0563 |
| KGSEQ | `one_to_one` | CE | 0.1039 | 0.0908 |
| KGSEQ | `epoch` | CE | 0.1039 | 0.0914 |

Cross-Entropy provides substantially better NDCG@20 than BPR and was therefore
selected for the final Fashion experiments.

The final five-seed CE results are:

| Model | Scheduling | Recall@20 | NDCG@20 |
|---|---|---:|---:|
| SASRec baseline | - | 0.0966 ± 0.00023 | 0.0889 ± 0.00016 |
| KGSEQ | `one_to_one` | **0.10366 ± 0.00044** | **0.09083 ± 0.00041** |
| KGSEQ | `epoch` | 0.10324 ± 0.00057 | 0.09032 ± 0.00065 |

Both alternate-learning schedules outperform the standalone SASRec baseline.

Paired Wilcoxon signed-rank tests between each KGSEQ configuration and the standalone SASRec baseline give:

```text
Recall@20: p = 0.0312
NDCG@20:   p = 0.0312
```

for both schedules.

The difference between `one_to_one` and `epoch` is not statistically
significant:

| Test | Recall@20 p | NDCG@20 p |
|---|---:|---:|
| one-sided | 0.0625 | 0.0938 |
| two-sided | 0.1250 | 0.1875 |

### Fashion-Random Relation Ablations

Selected Fashion-Random relation ablations were evaluated with both scheduling
strategies using a single seed.

| KG Configuration | Scheduling | Recall@20 | NDCG@20 |
|---|---|---:|---:|
| No brand | `epoch` | 0.1162 | 0.1063 |
| No brand | `one_to_one` | 0.1165 | 0.1065 |
| No category | `epoch` | 0.1235 | 0.1075 |
| No category | `one_to_one` | 0.1234 | 0.1079 |
| No `compatible_with` | `epoch` | 0.1209 | 0.1083 |
| No `compatible_with` | `one_to_one` | 0.1209 | 0.1086 |
| Full | `epoch` | 0.1235 | **0.1100** |
| Full | `one_to_one` | 0.1235 | 0.1099 |

These ablation results are single-seed experiments and should therefore be
interpreted as diagnostic comparisons rather than multi-seed estimates.

### Fashion-Heavy

Fashion-Heavy uses a separately tuned SASRec baseline because its user-selection
strategy produces a different interaction dataset from Fashion-Random.

The final five-seed results use `one_to_one` scheduling for alternate learning.

| Model | KG | Recall@20 | NDCG@20 |
|---|---|---:|---:|
| SASRec baseline | None | 0.09088 ± 0.00034 | 0.07734 ± 0.00023 |
| KGSEQ | 3 relations | **0.09796 ± 0.00028** | **0.08046 ± 0.00014** |
| KGSEQ | 5 relations | 0.09783 ± 0.00035 | 0.08041 ± 0.00008 |

Both KGSEQ configurations significantly outperform the standalone SASRec baseline.

For both the 3-relation and 5-relation configurations:

```text
Recall@20: p = 0.0312
NDCG@20:   p = 0.0312
```

The direct comparison between the 3-relation and 5-relation models is not
statistically significant:

```text
Recall@20: p = 0.8125
NDCG@20:   p = 0.6250
```

Therefore, adding the engineered price-tier and popularity-tier relations does
not provide a measurable downstream recommendation advantage.

### Fashion-Heavy Relation Ablations

Selected relation configurations were propagated from the isolated TransE
ablation study to the alternate-learning architecture.

These experiments use seed 2020 and `one_to_one` scheduling.

| KG configuration | KG family | Recall@20 | NDCG@20 |
|---|---|---:|---:|
| Full graph | 5-rel | 0.09786 | 0.08039 |
| Without `compatible_with` | 5-rel | 0.09384 | 0.07900 |
| Without price-tier and popularity-tier | 5-rel | 0.09766 | 0.08033 |
| `compatible_with` + `belongs_to_brand` | 5-rel | 0.09728 | 0.08024 |
| Full graph | 3-rel | 0.09822 | 0.08060 |
| Without `belongs_to` | 3-rel | 0.09692 | 0.08009 |

Removing the price-tier and popularity-tier relations improves isolated TransE
link-prediction performance over the full five-relation graph, but this advantage
does not transfer to downstream recommendation.
Removing `compatible_with` from the five-relation graph reduces Recall@20 from
0.09786 to 0.09384 and NDCG@20 from 0.08039 to 0.07900, corresponding to
relative reductions of approximately 4.1% and 1.7%, respectively.

These downstream ablations use a single seed and should therefore be interpreted
as diagnostic comparisons.

This result shows that improved isolated Knowledge Graph embedding performance
does not necessarily imply improved sequential recommendation performance.
## Reproducibility

Configuration files are stored under `configs/`.

Cluster-specific SLURM scripts are stored under `hpc/`. Paths in these scripts
may need to be adapted to the target HPC environment.

Experiment outputs such as checkpoints, Optuna databases, intermediate logs,
and resume checkpoints are generated artifacts and are not required as source
code for reproducing the pipeline.