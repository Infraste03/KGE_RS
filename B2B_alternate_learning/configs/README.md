# Configuration Files

This directory contains the YAML configuration files used for the B2B alternate learning experiments and for the stability evaluation of the standalone SASRec baseline.

The alternate learning architecture jointly optimizes:

- Task A: TransE-based Knowledge Graph link prediction
- Task B: SASRec-based sequential recommendation

Both components operate in a 400-dimensional latent space.

The configuration files define dataset paths, pretrained checkpoints, model architecture, training parameters, evaluation settings, and the alternate-learning scheduling strategy.


## Configuration Structure

The Step 4 configuration files are organized into four main sections.

### `paths`

Defines the resources required by the alternate learning pipeline:

- Knowledge Graph training triples
- Task A validation triples
- Task A test triples
- RecBole training interactions
- RecBole validation interactions
- RecBole test interactions
- pretrained TransE checkpoint
- pretrained SASRec checkpoint

These checkpoints are used to warm-start the two components before alternate training begins.


### `training`

Defines the optimization settings for the alternate learning procedure, including:

- KGE batch size
- SASRec batch size
- TransE margin loss
- learning rate
- number of epochs
- evaluation frequency
- number of negative samples for Task B
- scheduling strategy

The supported scheduling strategies include:

- `one_to_one`
- `epoch`
- `adaptive`

The final experiments reported in the project focus on the `one_to_one` and `epoch` strategies.


### `model`

Defines the architecture of the shared model.

The Knowledge Graph embedding dimension is fixed to:

`kge_dim: 400`

The SASRec architecture uses the same hidden dimension so that TransE and SASRec can directly operate on the same shared entity representations.

The remaining SASRec architecture parameters include:

- number of Transformer layers
- number of attention heads
- feed-forward inner dimension
- hidden dropout
- attention dropout


### `recbole_config`

Contains the RecBole settings required for Task B evaluation.

The evaluation protocol uses:

- temporal ordering
- user grouping
- full-ranking evaluation
- Recall@20
- NDCG@20

The main validation metric is `NDCG@20`.


## Files

### `step4_config.yaml`

Local/development configuration for the Step 4 alternate learning pipeline.

This configuration was mainly used for local testing and debugging.

It contains local Windows paths for the pretrained TransE and SASRec checkpoints and is configured for a short training run.

The current configuration uses:

- embedding dimension: 400
- KGE batch size: 512
- SASRec batch size: 512
- TransE margin: 5.19
- learning rate: 1e-4
- 4 negative recommendation samples
- `epoch` scheduling




### `step4_hpc_config.yaml`

Base configuration used to execute Step 4 on the HPC infrastructure.

Unlike the local configuration, it uses relative paths so that the experiment can be launched from the repository root.

It defines a 50-epoch alternate-learning experiment using:

- 400-dimensional shared embeddings
- 4 SASRec Transformer layers
- 2 attention heads
- KGE batch size: 512
- SASRec batch size: 512
- learning rate: 1e-4
- TransE margin: 5.19
- 4 negative samples for Task B
- `one_to_one` scheduling

This file represents the generic HPC configuration before replacing the training parameters with the configurations selected through HPO.


### `step4_seeds_one_to_one.yaml`

Final configuration selected by the HPO procedure for the `one_to_one` scheduling strategy.

The selected configuration was obtained from the HPO experiment with seed 2020.

The corresponding validation performance reported during model selection was:

- Recall@20: 0.5000
- NDCG@20: 0.2973
- selected epoch: 31

The selected hyperparameters are:

- KGE batch size: 1024
- SASRec batch size: 128
- learning rate: 8.108971322919983e-05
- TransE margin: 7.261743675271241
- Task B negative samples: 1
- hidden dropout: 0.44903633026041134
- attention dropout: 0.42532228700964586
- scheduling: `one_to_one`

This configuration is used for the final multi-seed experiments.

The YAML contains seed 2020 as a default value, but the seed can be overridden when launching individual runs.


### `step4_seeds_epoch.yaml`

Final configuration selected by the HPO procedure for the `epoch` scheduling strategy.

The corresponding validation performance reported during model selection was:

- Recall@20: 0.5093
- NDCG@20: 0.3053
- selected epoch: 21

The selected hyperparameters are:

- KGE batch size: 2048
- SASRec batch size: 1024
- learning rate: 0.0002214751193432723
- TransE margin: 1.9092406647496916
- Task B negative samples: 1
- hidden dropout: 0.26882517917076154
- attention dropout: 0.5122107440836169
- scheduling: `epoch`

As for the `one_to_one` configuration, the seed stored in the YAML acts as the default and can be overridden for the final stability experiments.


### `sasrec400_seeds.yaml`

Configuration used to retrain the standalone 400-dimensional SASRec baseline across multiple random seeds.

The architecture and training parameters reproduce the selected standalone SASRec checkpoint used as the Task B initialization.

The configuration includes:

- hidden dimension: 400
- 4 Transformer layers
- 2 attention heads
- inner dimension: 256
- hidden dropout: 0.4
- attention dropout: 0.3
- maximum sequence length: 50
- Cross-Entropy loss
- learning rate: 0.00031622776601683794
- weight decay: 3.1622776601683795e-05
- training batch size: 2048
- evaluation batch size: 4096
- maximum training epochs: 200

The seed specified in the YAML is only a default value and is overridden when running the different stability seeds.


## Recommended Experiment Configurations

For the final alternate-learning experiments, the reference configurations are:

`step4_seeds_one_to_one.yaml`

and

`step4_seeds_epoch.yaml`

These contain the hyperparameters selected independently through HPO for the two scheduling strategies.

`step4_config.yaml` and `step4_hpc_config.yaml` are retained primarily as development and base experiment configurations.

`sasrec400_seeds.yaml` is separate from the alternate-learning optimization and is used to evaluate the stability of the standalone SASRec baseline.