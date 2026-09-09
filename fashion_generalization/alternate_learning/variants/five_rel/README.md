# Fashion 5-Relation KG Variant

This folder contains the Fashion alternate-learning variant based on the complete 5-relation Knowledge Graph.

The original Fashion KG contains three relations:

- `compatible_with`
- `belongs_to`
- `belongs_to_brand`

The extended KG adds two additional relations:

- `belongs_to_price_tier`
- `belongs_to_pop_tier`

The corresponding data are stored under:

```text
fashion_generalization/data/processed_v2/
```

## Files

### `load_kg_fashion_5rel.py`

Loads the 5-relation Fashion KG and builds deterministic entity and relation mappings.

It supports the following entity types:

- `item`
- `category`
- `brand`
- `pricetier`
- `poptier`

### `unified_id_space_fashion_5rel.py`

Builds the unified ID space shared by the KG and RecBole.

It maps Fashion items between:

```text
KG entity IDs <-> Unified IDs <-> RecBole internal IDs
```

The standalone smoke test uses the 5-relation KG stored in `data/processed_v2/`.

### `data_loaders_fashion_5rel.py`

Provides the data loaders required by alternate learning for the 5-relation setting.

It prepares:

- Task A batches for TransE training;
- Task B sequential recommendation batches for SASRec.

### `run_step4_full5rel_fashion.py`

Runs Step 4 alternate learning using the complete 5-relation KG.

The model combines:

- Task A: TransE;
- Task B: SASRec with Cross-Entropy loss;
- a shared item embedding space.

The pretrained TransE checkpoint is expected at:

```text
results/hpo_transe_full5rel_v2/best_model.pt
```

The pretrained SASRec checkpoint is read from:

```text
alternate_learning/configs/step4_hpc_config_fashion_ce.yaml
```

through the `sasrec_checkpoint` configuration field.

## Running

From the `fashion_generalization` directory:

```bash
python alternate_learning/variants/five_rel/run_step4_full5rel_fashion.py
```

A scheduling strategy can also be specified:

```bash
python alternate_learning/variants/five_rel/run_step4_full5rel_fashion.py --scheduling epoch
```

Experiment outputs are stored under:

```text
alternate_learning/hpc_step4_results/
```