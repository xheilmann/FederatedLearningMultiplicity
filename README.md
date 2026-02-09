# Rashomon Sets and Model Multiplicity in Federated Learning

The Rashomon set captures the collection of models that achieve near-identical empirical performance yet may differ substantially in their decision boundaries. Understanding the differences among these models, i.e., their multiplicity, is recognized as a crucial step toward model transparency, fairness, and robustness, as it reveals decision boundaries instabilities that standard metrics obscure. 
However, the existing definitions of Rashomon set and multiplicity metrics assume centralized learning and do not extend naturally to decentralized, multi-party settings like Federated Learning (FL). In FL, multiple clients collaboratively train models under a central server’s coordination without sharing raw data, which preserves privacy but introduces challenges from heterogeneous client data distribution and communication constraints. 
In this setting, the choice of a single ``best'' model may homogenize predictive behavior across diverse clients, amplify biases, or undermine fairness guarantees.
In this work, we provide the first formalization of Rashomon sets in FL.
First, we adapt the Rashomon set definition to FL, distinguishing among three perspectives: (I) a \textbf{global} Rashomon set defined over aggregated statistics across all clients, (II) a \textbf{$t$-agreement} Rashomon set representing the intersection of local Rashomon sets across a fraction $t$ of clients, and (III) \textbf{individual} Rashomon sets specific to each client’s local distribution.
Second, we show how standard multiplicity metrics can be estimated under FL’s privacy constraints. 
Finally, we introduce a multiplicity‑aware FL pipeline and conduct an empirical study on standard FL benchmark datasets. Our results demonstrate that all three proposed federated Rashomon set definitions offer valuable insights, enabling clients to deploy models that better align with their local data, fairness considerations, and practical requirements.
## Code Features

- **Federated Learning Simulations for Multiplicity Research**: Implements a multiplicity-aware Federated Learning pipeline: Retraining strategy with subsequent multiplicity evaluation.
- **Dataset Support**: Built-in support for:
  - MNIST (classification, image data). 
  - Dutch (classification, tabular with sensitive attributes). This is available in /data/dutch as a CSV file.
  - ACS Income (classification, tabular with sensitive attributes). For this dataset, in the /data/income_reduced folder you can find a smaller version of the [original one](https://arxiv.org/abs/2108.04884).
- **Partitioning**: IID (uniform) and non-IID (Dirichlet-based) data partitioning.
- **Settings**: Cross-device (simulated clients) and cross-silo (realistic node setups).
- **Metrics and Logging**: Aggregates losses, accuracy (classification), RMSE/MAE/R2 (regression),rashomon capacity, disagreement, viable prediction range, discrepancy, ambiguity  and logs to WandB.
- **Extensibility**: Modular structure for custom models, strategies, and datasets.

## Before You Start

This project uses uv to manage dependencies. Make sure you have it installed. You can install it via pip:

```bash
pipx install uv
```

or using curl:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or wget:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Fore more details, visit the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Basic Usage

If you want to apply the retraining strategy to learn candidate models for the Rashomon set, you can run:

```bash
uv python  main.py --FL_setting cross_silo --num_clients 20 --dataset_name income --num_rounds 8 --sampled_train_nodes_per_round \
            0.25 --sampled_test_nodes_per_round 0.25 --sampled_validation_nodes_per_round 0.0 --node_shuffle_seed 42 --batch_size 38 --lr 0.0726338578312667 \
             --optimizer adam --momentum 0.5373415482935225  --fed_dir training_data/income/ --project_name Income_Models_multiplicity_2011 --run_name 8_20_3 --wandb True\
              --dataset_path data/income_reduced/ --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --num_epochs 1 --weight_decay 0.000418605274818734
```

## Basic Usage

If you want to try out a multiplicity evaluation, you can run:

```bash
uv python  multiplicity_evaluate.py --num_clients 20 --sampled_test_nodes_per_round 0.05 --dataset_name income --fed_dir training_data/income/  --wandb True --dataset_path data/income_reduced/ --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name individual_1 --project_name IncomeMultiplicityEvaluation --baseline_accuracy 0.75881 --baseline_model training_data/income/4_0.75_2
```

This will run a single multiplicity evaluation on the Dutch dataset with specified hyperparameters. Make sure that the models to be evaluated are saved in the path specified by the fed_dir parameter.
Make sure you have the dataset in the specified path and you have set up WandB if you want to log the results.

## Pipeline Usage
To run the complete pipeline, we propose running the run_{dataset}.py files first to train Rashomon set candidate models through the retraining strategy and then the run_{dataset}_multiplicity.py files to evaluate the models with respect to multiplicity.

## Configuration


Key parameters:
- `--num_clients`: Number of clients involved in the simulation.
- `--num_rounds`: Number of FL rounds that will be performed.
- `--num_epochs`: Local epochs per client.
- `--batch_size`: Local batch size for training.
- `--lr`: Learning rate.
- `--optimizer`: Optimizer type ("sgd", "adam").
- `--momentum`: Momentum for SGD.
- `--weight_decay`: Weight decay (L2 regularization).
- `--FL_setting`: "cross_device" or "cross_silo".
- `--dataset_name`: "mnist", "dutch", "income".
- `--partitioner_type`: "iid" or "non_iid" (with `--partitioner_alpha` for Dirichlet alpha).
- `--partitioner_alpha`: Dirichlet alpha parameter (float).
- `--partitioner_by`: Attribute to partition by (str).
- `--num_train_nodes`: Number of training nodes/clients. This is used in the cross-device setting.
- `--num_validation_nodes`: Number of validation nodes/clients. This is used in the cross-device setting.
- `--num_test_nodes`: Number of test nodes/clients. This is used in the cross-device setting.
- `--sampled_train_nodes_per_round`: Fraction of clients for training per round. E.g., 0.1 means 10% of clients.
- `--sampled_validation_nodes_per_round`: Fraction of clients for validation per round.
- `--sampled_test_nodes_per_round`: Fraction of clients for testing per round.
- `--fed_dir`: Directory where the results, logs and files related to the federated learning experiment will be saved.
- `--dataset_path`: Path to the dataset file (CSV for tabular, folder for images).
- `--sweep`: Enable hyperparameter sweep
- `--wandb`: Enable WandB logging
- `--project_name`: WandB project name.
- `--run_name`: WandB run name.
- `--task`: "classification" or "regression".
- `--image_path`: Path to the folder containing the images (for MNIST).
- `--baseline_model`: Path to the model serving as baseline.
- `-baseline_accuracy`: Accuracy of the above model.


## Acknowledgments

- Built with [Flower](https://flower.ai/).
- Datasets from UCI and Hugging Face.
- Logging with Weights & Biases.