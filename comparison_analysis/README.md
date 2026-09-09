# Dataset Comparison Analysis

This directory contains the analyses used to compare the two recommendation domains considered in this project:

- the industrial B2B spare-parts recommendation dataset;
- the Amazon Clothing, Shoes and Jewelry dataset used for the Fashion generalization experiments.

The purpose of this analysis is to quantify the structural differences between the two domains at both the interaction and Knowledge Graph levels.

The Fashion experiments are not intended to reproduce an identical B2B scenario on another dataset. Instead, they are designed as a cross-domain generalization experiment, in which the same general alternate-learning idea is evaluated under substantially different user behavior, sequence characteristics, catalog size, and Knowledge Graph semantics.

A second analysis compares the complete raw Amazon Fashion dataset with the 50,000-user subset used in the experiments, documenting the effect and coverage of the sampling procedure.


# Directory Structure

```text
comparison_analysis/
├── README.md
├── compare_datasets.py
├── compare_fashion_raw_vs_sample.py
│
├── b2b_vs_fashion/
│   ├── comparison_full_stats.json
│   ├── interaction_summary.csv
│   ├── kg_summary.csv
│   ├── kg_relation_stats.csv
│   ├── kg_entity_type_stats.csv
│   ├── kg_split_stats.csv
│   └── inter_kg_overlap.csv
│
└── fashion_raw_vs_sample/
    ├── raw_vs_sample_full.json
    ├── raw_vs_sample_summary.csv
    ├── sample_coverage.csv
    ├── metadata_summary.csv
    └── metadata_fields.csv
```


# 1. B2B vs Fashion Comparison

The script:

```text
compare_datasets.py
```

compares the interaction datasets and Knowledge Graphs used in the two experimental domains.

At the interaction level, the analysis computes:

- number of users;
- number of items;
- number of interactions;
- sparsity and density;
- sequence-length statistics;
- sequence-length Gini coefficient;
- repeated-purchase ratio;
- item-popularity statistics;
- item-popularity Gini coefficient;
- basket-size statistics;
- percentage of multi-item baskets.

At the Knowledge Graph level, the analysis computes:

- number of entities;
- number of relations;
- number of triples;
- number of duplicate triples;
- entity-type distributions;
- relation distributions;
- head and tail degree statistics;
- relation-specific degree Gini coefficients;
- KG train/validation/test split statistics;
- overlap between recommendation items and KG items.


# 2. Interaction Dataset Comparison

The two recommendation datasets differ substantially in both scale and interaction behavior.

| Statistic | B2B | Fashion |
|---|---:|---:|
| Users / Clients | 219 | 50,000 |
| Items | 21,134 | 165,469 |
| Interactions | 285,919 | 451,012 |
| Sparsity (%) | 93.82 | 99.99 |
| Average sequence length | 1,305.57 | 9.02 |
| Sequence-length Gini | 0.826 | 0.311 |
| Repeat purchase (%) | 74.47 | ~0 |
| Multi-item basket (%) | 74.93 | 33.02 |
| Item-popularity Gini | 0.824 | 0.525 |


## 2.1 User Population

The B2B dataset contains only 219 clients, whereas the Fashion subset contains 50,000 users.

The B2B scenario therefore represents a relatively small customer population with very rich historical purchasing information.

Fashion represents almost the opposite setting: a much larger population of users associated with considerably shorter interaction histories.


## 2.2 Sequence Length

One of the largest differences between the datasets concerns the amount of historical information available for each user.

The average sequence lengths are:

```text
B2B:     1,305.57
Fashion:     9.02
```

The median sequence lengths are:

```text
B2B:     144
Fashion:   7
```

The maximum observed sequence lengths are:

```text
B2B:     39,257
Fashion:    188
```

The B2B dataset therefore contains extremely long purchasing histories for some industrial clients.

The Fashion dataset instead contains many users with relatively short histories.

This difference is also visible in the sequence-length Gini coefficient:

```text
B2B:     0.826
Fashion: 0.311
```

The high B2B value indicates a strong concentration of activity: a relatively small subset of clients is responsible for a large portion of the interactions.

Fashion user activity is substantially more homogeneous.


## 2.3 Repeated Purchases

Repeated purchasing behavior is one of the strongest behavioral differences between the two domains.

```text
B2B repeat ratio:     74.47%
Fashion repeat ratio: ~0%
```

Repeated purchases are natural in an industrial spare-parts scenario, where the same component may be ordered multiple times because of maintenance, replacement, or recurring operational needs.

In the Fashion subset, repeated user-item interactions are essentially absent.

This results in two substantially different sequential recommendation problems.

In B2B, previous purchases may strongly indicate future recurring demand.

In Fashion, recommendation primarily involves ranking items that have not previously appeared in the user's sequence.


## 2.4 Basket Structure

The percentage of user-timestamp groups containing more than one item is:

```text
B2B:     74.93%
Fashion: 33.02%
```

The B2B dataset therefore exhibits a strong basket-oriented purchasing structure.

Industrial orders often contain multiple spare parts purchased together.

Multi-item events are also present in Fashion, but they are considerably less dominant.


## 2.5 Item Popularity

Item popularity is more concentrated in the B2B domain.

The item-popularity Gini coefficients are:

```text
B2B:     0.824
Fashion: 0.525
```

The B2B value indicates that purchasing activity is strongly concentrated around a relatively small subset of frequently requested components.

Fashion has a less concentrated popularity distribution and a broader long-tail structure.


## 2.6 Sparsity

Despite the much larger number of Fashion interactions, the Fashion user-item matrix is considerably more sparse:

```text
B2B sparsity:     93.82%
Fashion sparsity: 99.99%
```

This is mainly caused by the combination of:

- a much larger number of users;
- a much larger catalog;
- short user histories.

The B2B interaction matrix is denser because a relatively small number of clients repeatedly interact with the available spare-parts catalog.


# 3. Knowledge Graph Comparison

The two domains also differ substantially in their Knowledge Graph structure.

| Statistic | B2B | Fashion |
|---|---:|---:|
| Entities | 24,900 | 182,983 |
| Relations | 5 | 3 |
| Triples | 219,069 | 295,907 |

Although the Fashion Knowledge Graph contains considerably more entities, the B2B Knowledge Graph contains a richer set of relation types.


# 4. B2B Knowledge Graph

The B2B Knowledge Graph contains four main entity types:

```text
Clients
Items
Machines
Models
```

and five relations:

```text
owns
compatible_with
bought
instance_of
compatible_with_model
```

Their semantic structure is:

```text
owns
Client -> Machine

compatible_with
Item -> Machine

bought
Client -> Item

instance_of
Machine -> Model

compatible_with_model
Item -> Model
```

The `bought` relation derives from purchasing behavior, while the remaining structural relations describe the industrial environment in which the spare parts are used.


## 4.1 B2B Compatibility Semantics

A distinctive characteristic of the B2B Knowledge Graph is that compatibility has a strong physical and structural meaning.

For example, an item may be physically compatible with a specific machine or machine model.

Therefore, compatibility can often be interpreted approximately as a binary condition:

```text
compatible
vs.
not compatible
```

A corrupted compatibility triple may consequently represent an actual violation of the industrial structure.

This provides a relatively strong and explicit relational signal to the Knowledge Graph embedding model.


# 5. Fashion Knowledge Graph

The Fashion Knowledge Graph contains three main entity types:

```text
Items
Brands
Categories
```

and three relations:

```text
also_buy
belongs_to_brand
belongs_to_category
```

Their semantic structure is:

```text
also_buy
Item -> Item

belongs_to_brand
Item -> Brand

belongs_to_category
Item -> Category
```


## 5.1 Fashion Compatibility Semantics

The `also_buy` relation provides the main item-to-item compatibility signal in the Fashion Knowledge Graph.

However, its semantics are substantially different from the physical compatibility available in the B2B domain.

In Fashion:

```text
compatibility ≈ co-purchase signal
```

An `also_buy` edge indicates that two items have been behaviorally associated through co-purchase information.

This represents a softer and more semantic notion of compatibility.

The absence of an `also_buy` relation does not imply that two Fashion items are incompatible.

This differs from the B2B scenario, where compatibility relations can encode explicit structural or physical constraints.


# 6. Main Structural Differences Between the Knowledge Graphs

The two Knowledge Graphs can therefore be summarized as follows.

### B2B

```text
4 entity types
5 relation types
strong structural relations
physical compatibility
industrial domain knowledge
explicit machine-item relationships
small but heterogeneous entity space
```

### Fashion

```text
3 entity types
3 relation types
larger entity space
brand and category information
behavioral co-purchase relationships
soft semantic compatibility
item-centered graph structure
```

The Fashion Knowledge Graph is therefore not simply a larger version of the B2B graph.

It provides a different type of relational supervision.


# 7. Item Coverage Between Interaction Data and Knowledge Graph

The analysis also verifies whether items appearing in the recommendation dataset are represented in the corresponding Knowledge Graph.


## 7.1 B2B

```text
Interaction items:              21,134
KG items:                       21,077
Items shared by Inter and KG:   21,077
Items only in interactions:         57
Items only in KG:                    0
KG coverage of interaction items: 99.73%
```

Therefore, 57 B2B recommendation items are present in the interaction dataset but do not appear in the original Knowledge Graph.

These items are handled by the unified ID space of the alternate-learning architecture, which extends the shared entity space to include recommendation-only items.


## 7.2 Fashion

```text
Interaction items:             165,469
KG items:                      165,469
Items shared by Inter and KG: 165,469
Items only in interactions:          0
Items only in KG:                    0
KG coverage:                      100%
```

All Fashion recommendation items are represented in the Knowledge Graph.


# 8. Architectural Implication

The domain differences also lead to different representation dimensionalities in the two implementations.

The B2B alternate-learning architecture uses:

```text
shared / KGE embedding dimension = 400
```

The Fashion architecture uses:

```text
shared / KGE embedding dimension = 64
```

The Fashion dimensionality follows the representation size selected for the Fashion SASRec model.

This allows Task A and Task B to operate in the same shared latent space without requiring an additional projection layer.

The dimensionality difference should therefore be interpreted as an adaptation of the architecture to the target domain and its optimized sequential model, rather than as an attempt to keep all hyperparameters identical between B2B and Fashion.


# 9. Fashion Raw Dataset vs Experimental Subset

The script:

```text
compare_fashion_raw_vs_sample.py
```

analyzes the complete raw Amazon Clothing, Shoes and Jewelry dataset and compares it with the 50,000-user subset used in the experiments.

The analysis considers:

- raw review data;
- sampled interaction data;
- raw product metadata;
- user coverage;
- item coverage;
- interaction coverage;
- metadata coverage;
- category coverage;
- `also_buy` coverage.


# 10. Raw vs Sampled Fashion Dataset

| Statistic | Raw | Sample |
|---|---:|---:|
| Users | 12,483,678 | 50,000 |
| Items | 2,681,297 | 165,469 |
| Interactions | 32,292,099 | 451,012 |
| Sparsity (%) | 99.9999 | 99.9945 |
| Average sequence length | 2.59 | 9.02 |
| Median sequence length | 1 | 7 |
| Maximum sequence length | 656 | 188 |
| Repeat interactions (%) | 1.95 | ~0 |
| Items represented in metadata (%) | - | 100 |
| Items with category (%) | - | 100 |
| Items with `also_buy` (%) | - | 60.92 |


# 11. Why a Fashion Subset Is Used

The complete Amazon Fashion dataset is very large:

```text
12,483,678 users
2,681,297 items
32,292,099 interactions
```

However, the average amount of sequential information available for each user is very limited.

For the raw dataset:

```text
Average sequence length: 2.59
Median sequence length:  1
```

A median sequence length of one means that a very large portion of users provides almost no historical sequence from which a sequential recommendation model can learn.

The experimental subset was therefore constructed to retain users with sufficiently informative interaction histories.

The resulting sample contains:

```text
50,000 users
165,469 items
451,012 interactions
```

and has substantially longer histories:

```text
Average sequence length:
2.59 -> 9.02

Median sequence length:
1 -> 7
```

This makes the subset substantially more suitable for evaluating a sequential recommendation model such as SASRec.


# 12. Sample Coverage

Relative to the complete raw Fashion dataset, the experimental subset represents approximately:

| Measure | Coverage |
|---|---:|
| Users over raw dataset (%) | 0.40 |
| Items over raw dataset (%) | 6.17 |

Although only a small percentage of raw users is retained, the sample still contains a large and diverse item catalog.

The 50,000-user subset contains 165,469 unique items, which is substantially larger than the 21,134-item B2B catalog.


# 13. Metadata Coverage

Metadata availability was explicitly checked for all items retained in the Fashion subset.

The resulting coverage is:

```text
Items represented in metadata: 100%
Items with category:            100%
Items with also_buy:             60.92%
```

All sampled items can therefore be linked to metadata and category information.

The lower `also_buy` coverage is expected because not every Amazon item has an explicit co-purchase relationship.

This distinction is important for interpreting the Fashion Knowledge Graph.

Every item can potentially receive information from brand and category relations, while only a subset of the items receives explicit item-to-item information through `also_buy`.


# 14. Scientific Interpretation

The comparison shows that Fashion represents a substantial domain shift from the original B2B scenario.

The differences involve multiple dimensions simultaneously:

```text
number of users
catalog size
sequence length
repeat purchasing behavior
basket structure
interaction sparsity
popularity concentration
Knowledge Graph size
Knowledge Graph schema
compatibility semantics
embedding dimensionality
```

The Fashion dataset therefore provides a meaningful generalization scenario for evaluating the alternate-learning architecture.


## B2B Scenario

The B2B domain is characterized by:

```text
few clients
very long interaction histories
high repeated-purchase behavior
strong multi-item purchasing
higher popularity concentration
physical compatibility constraints
heterogeneous industrial entities
strong structural KG information
```


## Fashion Scenario

The Fashion domain is characterized by:

```text
many users
short interaction histories
almost no repeated user-item purchases
very large item catalog
extremely sparse interaction matrix
lower popularity concentration
soft co-purchase compatibility
brand and category information
larger but structurally simpler Knowledge Graph
```


# 15. Implication for the Generalization Experiment

Because the two domains are structurally different, absolute recommendation metric values should not be directly compared across B2B and Fashion.

For example, a Recall@20 value obtained on B2B and a Recall@20 value obtained on Fashion correspond to substantially different:

- candidate spaces;
- numbers of users;
- sequence-length distributions;
- item catalogs;
- interaction densities;
- purchasing behaviors.

The relevant comparison is therefore performed within each domain.

The main question is whether the knowledge-enhanced alternate-learning architecture improves over the corresponding standalone sequential baseline under each experimental setting.

The Fashion experiment consequently evaluates whether the benefit of integrating sequential recommendation and Knowledge Graph representations persists when moving from:

```text
industrial spare-parts recommendation
```

to:

```text
consumer Fashion recommendation
```

under substantially different data and KG conditions.


# 16. Generated Files

## `b2b_vs_fashion/comparison_full_stats.json`

Contains the complete nested output of the B2B vs Fashion analysis, including interaction, KG, and item-overlap statistics.


## `b2b_vs_fashion/interaction_summary.csv`

Contains the main interaction statistics for both datasets, including:

```text
number of users
number of items
number of interactions
sparsity
density
repeat ratio
sequence-length statistics
sequence-length Gini
item-popularity statistics
item-popularity Gini
multi-item basket percentage
```


## `b2b_vs_fashion/kg_summary.csv`

Contains the main Knowledge Graph statistics:

```text
number of entities
number of relations
number of triples
duplicate triples
unique heads
unique tails
```


## `b2b_vs_fashion/kg_relation_stats.csv`

Contains relation-level statistics, including:

```text
number of triples
percentage of total triples
number of unique heads
number of unique tails
head and tail entity types
head-degree statistics
tail-degree statistics
degree Gini coefficients
```


## `b2b_vs_fashion/kg_entity_type_stats.csv`

Contains the number of entities associated with each entity type.


## `b2b_vs_fashion/kg_split_stats.csv`

Contains the number of triples associated with each KG split file.


## `b2b_vs_fashion/inter_kg_overlap.csv`

Contains the overlap between the items appearing in the recommendation dataset and those represented in the corresponding Knowledge Graph.


## `fashion_raw_vs_sample/raw_vs_sample_full.json`

Contains the complete machine-readable output of the Fashion raw-vs-sample analysis.


## `fashion_raw_vs_sample/raw_vs_sample_summary.csv`

Contains the main statistics of the complete raw Fashion dataset and the experimental subset.


## `fashion_raw_vs_sample/sample_coverage.csv`

Contains the percentage of users, items, and interactions retained from the raw Fashion dataset.


## `fashion_raw_vs_sample/metadata_summary.csv`

Contains metadata coverage information for the items included in the experimental subset.


## `fashion_raw_vs_sample/metadata_fields.csv`

Contains statistics about the fields available in the raw Amazon Fashion metadata.


# 17. Running the Analyses

The scripts can be executed from the repository root.

To run the B2B vs Fashion comparison:

```bash
python comparison_analysis/compare_datasets.py
```

To run the Fashion raw-vs-sample comparison:

```bash
python comparison_analysis/compare_fashion_raw_vs_sample.py
```

The scripts derive their paths from the repository structure and therefore do not require machine-specific absolute paths.


# 18. Dependencies

The analyses require:

```text
Python
pandas
numpy
```

The Fashion raw-vs-sample analysis also uses the Python standard-library modules:

```text
gzip
json
collections
pathlib
```


# 19. Reproducibility

The Python scripts are the source of truth for reproducing the comparison analyses.

The generated CSV and JSON files are retained in the repository because they provide:

- a transparent record of the dataset statistics used in the experimental analysis;
- direct access to the results without processing the complete raw datasets again;
- reproducible values for tables and presentation material;
- documentation of the structural differences considered when interpreting the cross-domain generalization experiments.

The comparison results should therefore be interpreted as part of the experimental documentation of the project, rather than as independent model results.