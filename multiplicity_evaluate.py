import argparse
import os
import signal
import time
from typing import Any

import numpy as np
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr_datasets.visualization import plot_label_distributions

from Aggregations.aggregations import Aggregation
from ClientManager.client_manager import SimpleClientManager
from datasets import load_dataset
from Datasets.dataset_utils import get_data_info, prepare_data_for_cross_device, prepare_data_for_cross_silo
from main import setup_wandb, signal_handler
from Models.utils import get_model
from Server.evaluation_server import EvalServer
from Strategy.custom_evaluation import CustomEvaluation
from Utils.preferences import Preferences
from Utils.utils import seed_everything


def client_fn(context: Context) -> Any:
    """
    Generates a Flower client instance with its assigned data partition.

    Loads the partition based on the global partitioner and prepares data for the specified FL setting (cross-device or cross-silo).

    Args:
        context (Context): The Flower context with node configuration including partition ID.

    Returns:
        Any: A configured Flower client instance.

    Raises:
        KeyError: If "partition-id" is not found in node_config.
    """
    partition_id = int(context.node_config["partition-id"])
    if partitioner:
        partition = partitioner.load_partition(partition_id)
    else:
        partition = None

    if preferences.cross_device:
        return prepare_data_for_cross_device(context, partition, preferences, partition_id)

    return prepare_data_for_cross_silo(context, partition, preferences, partition_id)

def server_fn(context: Context) -> ServerAppComponents:
    """
    Constructs ServerAppComponents for running the Flower server simulation.

    Initializes the global model, defines the FedAvg strategy with aggregation functions, and sets up the server with client manager and preferences.

    Args:
        context (Context): The Flower context.

    Returns:
        ServerAppComponents: Components including the server instance and configuration.
    """
    # # instantiate the models
    # model = get_model(dataset=preferences.dataset_name)
    # global_model_init={}
    # for dir in os.listdir(preferences.fed_dir):
    #     if os.path.isdir(os.path.join(preferences.fed_dir, dir)):
    #         ndarrays = numpy.load(f"{preferences.fed_dir}/{dir}/model.npz")
    # # Convert model parameters to flwr.common.Parameters
    #         global_model_init[dir] = ndarrays_to_parameters([ndarrays["arr_0"]])
    model = get_model(dataset=preferences.dataset_name)
    global_model_init = {}

    for dir_name in os.listdir(preferences.fed_dir):
        full_path = os.path.join(preferences.fed_dir, dir_name)

        if os.path.isdir(full_path):
            # Load the .npz file (acts like a dictionary)
            with np.load(f"{full_path}/model.npz") as data:
                # FIX: Extract ALL arrays stored in the file, not just arr_0
                # .files attribute lists keys like ['arr_0', 'arr_1', 'arr_2', 'arr_3']
                loaded_arrays = [data[key] for key in data.files]

                # Convert the list of all arrays to Flower Parameters
                global_model_init[dir_name] = ndarrays_to_parameters(loaded_arrays)

    print(f"Loaded parameters for {len(global_model_init)} models.")

    # Define the strategy
    strategy = CustomEvaluation(
        fraction_evaluate=preferences.sampled_validation_nodes_per_round,
        initial_parameters=global_model_init,  # initialised global model
        evaluate_metrics_aggregation_fn=Aggregation.agg_metrics_evaluation,
        preferences=preferences,
        wandb_run=wandb_run,
    )

    config = ServerConfig(num_rounds=num_rounds)
    server = EvalServer(client_manager=client_manager, strategy=strategy, preferences=preferences)

    # Wrap everything into a `ServerAppComponents` object
    return ServerAppComponents(server=server, config=config)

def get_partitioner(preferences: Preferences) -> Any:
    """
    Returns a partitioner based on the specified type in preferences.
    Supports "iid" and "non_iid" (Dirichlet) partitioning.

    Args:
        preferences (Preferences): User preferences containing partitioner settings.

    Returns:
        Any: An instance of the selected partitioner.

    Raises:
        ValueError: If an unsupported partitioner type is specified.
    """
    partitioner_type = preferences.partitioner_type

    match partitioner_type:
        case "iid":
            return IidPartitioner(num_partitions=preferences.num_clients)
        case "non_iid":
            return DirichletPartitioner(
                num_partitions=preferences.num_clients,
                alpha=preferences.partitioner_alpha,
                partition_by=preferences.partitioner_by,
            )
        case _:
            error = f"Unsupported partitioner type: {partitioner_type}"
            raise ValueError(error)

def prepare_data(preferences: Preferences) -> Any:
    """
    Loads and prepares the dataset based on the specified name in preferences.

    Supports datasets: "dutch", "mnist", "abalone", "income". Sets up scaler/encoder if applicable, creates partitioner, and optionally plots label distributions.

    Args:
        preferences (Preferences): User preferences containing dataset settings.

    Returns:
        Any: The partitioner instance for data partitioning, or None for "income" dataset.

    Raises:
        ValueError: If an unsupported dataset is specified or no training data is found.
    """
    if preferences.dataset_name == "dutch":
        data_info = get_data_info(preferences)
        preferences.scaler = data_info.get("scaler", None)
        dataset_dict = load_dataset("csv", data_files=preferences.dataset_path)
    elif preferences.dataset_name == "mnist":
        data_info = get_data_info(preferences)
        dataset_dict = load_dataset(data_info["data_type"], data_dir=preferences.dataset_path)
    elif preferences.dataset_name == "abalone":
        data_info = get_data_info(preferences)
        preferences.scaler = data_info.get("scaler", None)
        dataset_dict = load_dataset("csv", data_files=preferences.dataset_path)
    elif preferences.dataset_name == "income":
        data_info = get_data_info(preferences)
        preferences.scaler = data_info.get("scaler", None)
        preferences.encoder = data_info.get("encoder", None)
        partitioner = None
        return partitioner
    elif preferences.dataset_name == "celeba":
        data_info = get_data_info(preferences)
        dataset_dict = load_dataset("csv", data_files=preferences.dataset_path)
    # elif preferences.dataset_name == "speech_fairness":

    else:
        error = f"Unsupported dataset: {preferences.dataset_name}"
        raise ValueError(error)

    data = dataset_dict.get("train", None)  # type: ignore
    if data:
        partitioner = get_partitioner(preferences)
        partitioner.dataset = data
    else:
        error = "No training data found in the dataset"
        raise ValueError(error)

    if args.partitioner_by:
        plot, _, _ = plot_label_distributions(
            partitioner=partitioner,
            label_name=args.partitioner_by,
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            max_num_partitions=args.num_clients,
            title="Per Partition Labels Distribution",
        )
        plot.savefig(f"label_distribution_{args.partitioner_by}_{args.partitioner_type}.png", bbox_inches="tight")

    return partitioner

parser = argparse.ArgumentParser(description="Flower Evaluation of Pre-trained Models")
parser.add_argument("--num_clients", type=int, default=None, required=True)
parser.add_argument("--dataset_name", type=str, default=None, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--node_shuffle_seed", type=int, default=None)
parser.add_argument("--partitioner_type", type=str, default="non_iid")
parser.add_argument("--partitioner_alpha", type=float, default=None)
parser.add_argument("--partitioner_by", type=str, default=None)
parser.add_argument("--fed_dir", type=str, default=None, required=True)
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--wandb", type=bool, default=True)
parser.add_argument("--project_name", type=str, default="FlowerFLTemplate")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--task", type=str, default="classification")
parser.add_argument("--sampled_test_nodes_per_round", type=float, default=1.0)
parser.add_argument("--threshold", type=float, default=None)
parser.add_argument( "--setting", type=str, default="multiplicity")
parser.add_argument("--baseline_model", type=str, required=True)
parser.add_argument("--baseline_accuracy", type=float, required=True)





if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    # remove files in tmp/ray
    args = parser.parse_args()

    if args.node_shuffle_seed is None:
        node_shuffle_seed = int(str(time.time()).split(".")[1]) * args.seed
        args.node_shuffle_seed = node_shuffle_seed
    seed_everything(args.seed)

    num_clients = args.num_clients
    num_rounds = 1

    cross_device = "cross_device"

    preferences = Preferences(
        num_clients=num_clients,
        num_rounds=num_rounds,
        cross_device=False,
        num_test_nodes=args.num_clients,
        sampled_test_nodes_per_round=args.sampled_test_nodes_per_round,
        seed=args.seed,
        node_shuffle_seed=args.node_shuffle_seed,
        fed_dir=args.fed_dir,
        sweep=args.sweep,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        partitioner_type=args.partitioner_type,
        partitioner_alpha=args.partitioner_alpha,
        partitioner_by=args.partitioner_by,
        task=args.task,
        num_validation_nodes=num_clients,
        num_train_nodes=num_clients,
        num_epochs=1,
        sampled_validation_nodes_per_round=1,
        sampled_training_nodes_per_round=1,
        threshold = args.threshold,
        setting= args.setting,
        baseline_model=args.baseline_model,
        baseline_accuracy = args.baseline_accuracy,

    )


    wandb_run = (
        setup_wandb(
            project_name=args.project_name,
            run_name=args.run_name,
        )
        if args.wandb
        else None
    )

    # Create your ServerApp
    client_manager = SimpleClientManager(preferences=preferences)

    partitioner = prepare_data(preferences=preferences)

    # Create your ServerApp
    server_app = ServerApp(server_fn=server_fn)

    # Concstruct the ClientApp passing the client generation function
    client_app = ClientApp(client_fn=client_fn)

    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=num_clients)

    if wandb_run:
        wandb_run.finish()
