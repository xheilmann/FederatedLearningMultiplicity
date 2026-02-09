import json
import os
import shutil
import shutil
from typing import Any, List

import numpy
import numpy as np
import torch
from datasets import load_dataset
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, Parameters
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from Datasets.mnist import prepare_mnist_for_centralised_evaluation

from Client.evaluation_client import load_models_sorted_by_epsilon, load_parameters_from_file
from Datasets.dutch import prepare_dutch, DutchDataset
from Datasets.income import get_income_scaler, prepare_income, IncomeDataset
from Models.regression_model import RegressionModel
from Models.simple_model import SimpleModel
from Models.utils import get_model
from torch import nn
from torch.utils.data import DataLoader
from Utils.multiplicity_evaluation import *

# from Training.training import test, train
from Utils.preferences import Preferences
from Utils.utils import get_optimizer, get_params, set_params
from main import setup_wandb

epsilons = [0.0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.024, 0.028, 0.032, 0.036, 0.04]

def load_parameters_from_file(file_path: str) -> NDArrays:
    """Load all parameters from a .npz file generically."""
    
    # numpy.load acts as a context manager, so we don't need 'open()'
    with np.load(file_path) as data:
        # data.files is a list of all keys in the file (e.g., ['arr_0', 'arr_1', ...])
        # We extract every array associated with these keys.
        parameters = [data[key] for key in data.files]
        
    return parameters

def centralized_evaluate(path, preferences, project_name=None, run_name=None):
    wandb_run = setup_wandb(
            project_name=project_name,
            run_name=run_name,
        )



    trained_model = get_model(dataset=preferences.dataset_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleModel(
            model=trained_model,
            optimizer=get_optimizer(trained_model, preferences),
            criterion=nn.CrossEntropyLoss(),
            device=device,
        )

    if preferences.dataset_name=="income":
        all_files = []
        for file_name in os.listdir(preferences.dataset_path):
            # check if the file is a folder
            if os.path.isdir(os.path.join(preferences.dataset_path, file_name)):
                for f in os.listdir(os.path.join(preferences.dataset_path, file_name)):
                    if f.endswith(".csv"):
                        all_files.append(os.path.join(preferences.dataset_path, file_name, f))
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        df, test = train_test_split(df, test_size=0.2)
        scaler, encoder = get_income_scaler(
            sweep=False,
            seed=42,
            df=test,
            validation_seed=10342,
        )

        x_test, z_test, y_test, _, _ = prepare_income(
            df=test,
            scaler=scaler,
            encoder=encoder,
        )

        test_dataset = IncomeDataset(
            x=np.hstack((x_test, np.ones((x_test.shape[0], 1)))).astype(np.float32),
            z=z_test.astype(np.float32),
            y=y_test.astype(np.float32),
        )

        print("Test dataset size:", len(test_dataset))

        valloader = DataLoader(test_dataset, shuffle=False)

    elif preferences.dataset_name=="dutch":

        dataset_dict = load_dataset("csv", data_files=preferences.dataset_path)
        test = dataset_dict.get("train").to_pandas()
        train, test = train_test_split(test, test_size=0.2)

        x_test, z_test, y_test, _ = prepare_dutch(
            dutch_df=test,
            scaler=preferences.scaler,
        )

        test_dataset = DutchDataset(
            x=np.hstack((x_test, np.ones((x_test.shape[0], 1)))).astype(np.float32),
            z=z_test.astype(np.float32),
            y=y_test.astype(np.float32),
        )

        print("Test dataset size:", len(test_dataset))
        valloader = DataLoader(test_dataset, shuffle=False)
    
    elif preferences.dataset_name == "mnist":
        print("Preparing MNIST dataset for centralized evaluation...")
        valloader = prepare_mnist_for_centralised_evaluation()
        print("Test dataset size:", len(valloader.dataset))


    global_model_init = {}
    result_dict = {}
    for dir in os.listdir(f"{path}"):
        if os.path.isdir(os.path.join(f"{path}", dir)):
            # ndarrays = numpy.load(f"{path}/{dir}/model.npz")
            parameters = load_parameters_from_file(f"{path}/{dir}/model.npz")
            # Convert model parameters to flwr.common.Parameters
            global_model_init[dir] = parameters
        
    print("Loaded models for evaluation:", list(global_model_init.keys()))
    
    print("Starting evaluation of models...")
    for key, value in global_model_init.items():
        set_params(model.model, value)
        evaluation = model.evaluate(testloader=valloader)
        result_dict[f"{key}_acc"] =evaluation["accuracy"]
        result_dict[f"{key}_dp_local"] = evaluation["local_dp"]
        result_dict[f"{key}_dp"] = json.dumps(evaluation["dp"])
        print(f"Evaluated model {key}: Accuracy = {evaluation['accuracy']}, Local DP = {evaluation['local_dp']}, DP = {evaluation['dp']}")
    print("Completed evaluation of all models.")


    # Create epsilon directories

    epsilon_counts = {epsilon: 0 for epsilon in epsilons}
    sorted_models = {epsilon: [] for epsilon in epsilons}
    sorted_models[None] = []


    processed_models = set()  # To ensure each model is only sorted once

    print("Sorting models into epsilon buckets...")
    for key, accuracy in result_dict.items():
        if not key.endswith("_acc"):
            continue

        model_name = key.replace("_acc", "")

        # Skip if we've already processed this model
        if model_name in processed_models:
            continue

        # Calculate difference from baseline
        if preferences.threshold is not None:
            diff = accuracy
        else:
            diff = abs(accuracy - preferences.baseline_accuracy)

        # Find appropriate epsilon range
        assigned_epsilon = None
        for epsilon in sorted(epsilons):
            if diff <= epsilon:
                assigned_epsilon = epsilon
                break


        if assigned_epsilon is None:
            #epsilon_counts[None] += 1
            bucket_key = None

        else:
            epsilon_counts[assigned_epsilon] += 1
            bucket_key = assigned_epsilon


        sorted_models[bucket_key].append(global_model_init[model_name])

        processed_models.add(model_name)

    rashomonratio = [float(epsilon_counts[eps]) for eps in sorted(epsilons)]
    
    print("Evaluating sorted models for multiplicity metrics...")
    global_model_init = {}
    result_dict = {}
    for dir in os.listdir(f"{path}"):
        if os.path.isdir(os.path.join(f"{path}", dir)):
            # ndarrays = numpy.load(f"{path}/{dir}/model.npz")
            parameters = load_parameters_from_file(f"{path}/{dir}/model.npz")
            # Convert model parameters to flwr.common.Parameters
            global_model_init[dir] = parameters
    for key, value in global_model_init.items():
        set_params(model.model, value)
        result_dict[f"{key}_acc"] = model.evaluate(testloader=valloader)["accuracy"]
        result_dict[f"{key}_dp_local"] = model.evaluate(testloader=valloader)["local_dp"]
        result_dict[f"{key}_dp"] = json.dumps(model.evaluate(testloader=valloader)["dp"])
        print(f"Evaluated sorted model {key}: Accuracy = {result_dict[f'{key}_acc']}, Local DP = {result_dict[f'{key}_dp_local']}, DP = {result_dict[f'{key}_dp']}")

            # Create base directory if it doesn't exist
    os.makedirs(f"Rashomon_Set_{preferences.dataset_name}", exist_ok=True)
    #remove the old Rashomon set directories
    for item in os.listdir(f"Rashomon_Set_{preferences.dataset_name}/"):
        item_path = os.path.join(f"Rashomon_Set_{preferences.dataset_name}/", item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)  # remove directory

    # Create epsilon directories
    epsilon_dirs = {}
    epsilon_counts = {}
    epsilon_counts[None] = 0
    for epsilon in sorted(epsilons):
        dir_name = f"Rashomon_Set_{preferences.dataset_name}/epsilon_{epsilon}"
        os.makedirs(dir_name, exist_ok=True)
        epsilon_dirs[epsilon] = dir_name
        epsilon_counts[epsilon] = 0

    # Process each model
    processed_models = set()  # To ensure each model is only sorted once

    for key, accuracy in result_dict.items():
        if not key.endswith("_acc"):
            continue

        model_name = key.replace("_acc", "")

        # Skip if we've already processed this model
        if model_name in processed_models:
            continue

        # Calculate difference from baseline
        if preferences.threshold is not None:
            diff = accuracy
        else:
            diff = abs(accuracy - preferences.baseline_accuracy)

        # Find appropriate epsilon range
        assigned_epsilon = None
        for epsilon in sorted(epsilons):
            if diff <= epsilon:
                assigned_epsilon = epsilon
                break

        # If no epsilon range matches, put in a "no_match" directory
        if assigned_epsilon is None:
            dir_name = f"Rashomon_Set_{preferences.dataset_name}/no_match"
            os.makedirs(dir_name, exist_ok=True)
            target_dir = dir_name
            epsilon_counts[None] += 1

        else:
            target_dir = epsilon_dirs[assigned_epsilon]
            epsilon_counts[assigned_epsilon] += 1


        processed_models.add(model_name)

    rashomonratio = [float(epsilon_counts[epsilon]) for epsilon in sorted(epsilons)]


    # Define base directory where models are stored
    base_model_params = load_parameters_from_file(preferences.baseline_model + "/model.npz")
    set_params(model.model, base_model_params)
    model.model.eval()
    base_pred_y = []
    for sample, _, label in valloader:
        images, labels = sample.to(device), label.to(device)
        outputs = model.model(images)
        base_pred_y.append(torch.max(outputs.data, 1)[1])

    base_pred_y = [x for xs in base_pred_y for x in xs]
    # Load the sorted models from directories
    # print(base_pred_y)

    len_eps_list = len(sorted_models.items())

    all_sampling_test_scores = []

    i = 0

    # Process models in epsilon order
    ntest = len(base_pred_y)
    vpr_difpriv = np.zeros((len_eps_list, 1000))
    score_var_difpriv = np.zeros((len_eps_list, 1000))
    rc_difpriv = np.zeros((len_eps_list, 1000))
    vpr = np.zeros((len_eps_list, ntest))
    score_var = np.zeros((len_eps_list, ntest))
    rc = np.zeros((len_eps_list, ntest))
    amb = np.zeros((len_eps_list,))
    disc = np.zeros((len_eps_list,))
    disa_hat_difpriv = np.zeros((len_eps_list, 1000))
    disa_hat = np.zeros((len_eps_list, ntest))
    for epsilon, model_params in sorted_models.items():
        for model_param in model_params:
            # Load model parameters
            sampling_test_scores = []
            set_params(model.model, model_param)
            model.model.eval()
            for sample, _, label in valloader:
                images, labels = sample.to(device), label.to(device)
                outputs = model.model(images)
                sampling_test_scores.append(outputs.data)
            all_sampling_test_scores.append([x for xs in sampling_test_scores for x in xs])

        if not all_sampling_test_scores:
            i += 1
            continue

        scores = score_of_y_multi_model(all_sampling_test_scores, base_pred_y)

        vpr_difpriv[i, :], vpr[i, :] = viable_prediction_range(scores)
        score_var_difpriv[i, :], score_var[i, :] = score_variance(scores)
        rc_difpriv[i,:], rc[i, :] = rashomon_capacity(all_sampling_test_scores)
        print(rc)

        decisions = [[torch.max(subsubarray, 0)[1] for subsubarray in subarray] for subarray in
                     all_sampling_test_scores]
        amb[i] = ambiguity(decisions, base_pred_y)
        disc[i] = discrepancy(decisions, base_pred_y)
        disa_hat_difpriv[i, :], disa_hat[i, :] = disagreement_hat(decisions)

        i += 1

    disa_hat_cum = np.cumsum(disa_hat_difpriv,  axis=1)
    disa_hat_cum = [entry / entry[-1] for entry in disa_hat_cum]

    disa_hat_90_difpriv = [float(np.searchsorted(entry, 0.9, "left") * 0.001) for entry in disa_hat_cum]
    disa_hat_50_difpriv = [float(np.searchsorted(entry, 0.5) * 0.001) for entry in disa_hat_cum]
    for entry in disa_hat_cum:
        entry[np.isnan(entry)] = 0


    disa_hat_90 = np.quantile(disa_hat, 0.9, axis=1)
    disa_hat_50 = np.quantile(disa_hat, 0.5, axis=1)


    vpr_cum = np.cumsum(vpr_difpriv, axis=1)
    vpr_cum = [entry / entry[-1] for entry in vpr_cum]

    vpr_90_difpriv = [float(np.searchsorted(entry, 0.9) * 0.001) for entry in vpr_cum]
    vpr_50_difpriv = [float(np.searchsorted(entry, 0.5) * 0.001) for entry in vpr_cum]
    for entry in vpr_cum:
        entry[np.isnan(entry)] = 0
    vpr_difpriv = vpr_cum



    vpr_90 = np.quantile(vpr, 0.9, axis=1)
    vpr_50 = np.quantile(vpr, 0.5, axis=1)


    score_var_cum = np.cumsum(score_var_difpriv, axis=1)
    score_var_cum = [entry / entry[-1] for entry in score_var_cum]
    # print(score_var, score_var_cum)

    score_var_90_difpriv = [float(np.searchsorted(entry, 0.9) * 0.001) for entry in score_var_cum]
    score_var_50_difpriv = [float(np.searchsorted(entry, 0.5) * 0.001) for entry in score_var_cum]
    for entry in score_var_cum:
        entry[np.isnan(entry)] = 0
    score_var_difpriv = score_var_cum
    # print(agg_metrics[f"score_var_90"], agg_metrics[f"score_var_50"])


    score_var_90 = np.quantile(score_var, 0.9, axis=1)
    score_var_50 = np.quantile(score_var, 0.5, axis=1)


    rc_cum = np.cumsum(rc_difpriv,axis=1)
    rc_cum = [entry / entry[-1] for entry in rc_cum]

    rc_90_difpriv = [float(np.searchsorted(entry, 0.9, "left") * 0.001) for entry in rc_cum]
    rc_50_difpriv = [float(np.searchsorted(entry, 0.5) * 0.001) for entry in rc_cum]
    for entry in rc_cum:
        entry[np.isnan(entry)] = 0
    rc_difpriv= rc_cum

    rc_90= np.quantile(rc, 0.9, axis=1)
    rc_50 = np.quantile(rc, 0.5, axis=1)



    savename = f'centralized_{preferences.dataset_name}_results.npz'
    np.savez_compressed(path + savename,
                        all_sampling_test_scores=all_sampling_test_scores,
                        eps_list=list(sorted_models.keys()),
                        vpr=vpr,
                        score_var=score_var,
                        amb=amb,
                        disc=disc,
                        disa_hat=disa_hat,
                        rc=rc,
                        rc_difpriv=rc_difpriv,
                        vpr_difpriv=vpr_difpriv,
                        score_var_difpriv=score_var_difpriv,
                        disa_hat_difpriv=disa_hat_difpriv  ,
                        disa_hat_90 = disa_hat_90,
                        disa_hat_50 = disa_hat_50,
                        disa_hat_90_difpriv = disa_hat_90_difpriv,
                        disa_hat_50_difpriv = disa_hat_50_difpriv,
                        vpr_90 = vpr_90,
                        vpr_50 = vpr_50,
                        vpr_90_difpriv = vpr_90_difpriv,
                        vpr_50_difpriv = vpr_50_difpriv,
                        score_var_90 = score_var_90,
                        score_var_50 = score_var_50,
                        score_var_90_difpriv = score_var_90_difpriv,
                        score_var_50_difpriv = score_var_50_difpriv,
                        rc_90 = rc_90,
                        rc_50 = rc_50,
                        rc_90_difpriv = rc_90_difpriv,
                        rc_50_difpriv = rc_50_difpriv
                        )

    wandb_run.log({
        "vpr": vpr,
        "score_var": score_var,
        "vpr_difpriv": vpr_difpriv,
        "score_var_difpriv": score_var_difpriv,
        "rc":rc,
        "rc_difpriv": rc_difpriv,
        "amb": amb,
        "disc": disc,
        "disa_hat": disa_hat,
        "disa_hat_difpriv": disa_hat_difpriv, "disa_hat_90": disa_hat_90, "disa_hat_50": disa_hat_50,
        "disa_hat_90_difpriv": disa_hat_90_difpriv, "disa_hat_50_difpriv": disa_hat_50_difpriv,
        "vpr_90": vpr_90, "vpr_50": vpr_50, "vpr_90_difpriv": vpr_90_difpriv, "vpr_50_difpriv": vpr_50_difpriv,
        "score_var_90": score_var_90, "score_var_50": score_var_50, "score_var_90_difpriv": score_var_90_difpriv,
        "score_var_50_difpriv": score_var_50_difpriv,
        "rc_90": rc_90, "rc_50": rc_50, "rc_90_difpriv": rc_90_difpriv, "rc_50_difpriv": rc_50_difpriv, 
        "rashomon_ratio": rashomonratio})
    wandb_run.log(result_dict)


    wandb_run.finish()

if __name__ == "__main__":



    # preferences = Preferences(dataset_name="dutch", dataset_path="data/dutch/dutch.csv", momentum=0.9, num_epochs=5, optimizer="adam", baseline_model="training_data/dutch_10_clients/7_0.25_2117271212/", baseline_accuracy=0.8358)

    # centralized_evaluate("training_data/dutch_10_clients/", preferences, project_name="DutchMultiplicityEvaluation_centralized", run_name="centralized")


    preferences = Preferences(dataset_name="mnist", dataset_path="./data/MNIST/train/", momentum=0.9, num_epochs=5, optimizer="adam", baseline_model="training_data/mnist/5_0.25_1/", baseline_accuracy=0.92167)

    centralized_evaluate("training_data/mnist/", preferences, project_name="MNISTMultiplicityEvaluation_centralized", run_name="centralized")


    # preferences = Preferences(dataset_name="income", dataset_path="./training_data/income_reduced/", momentum=0.9,  num_epochs=5, baseline_accuracy=0.75881, baseline_model="training_data/income/4_0.75_2", optimizer="adam")

    # centralized_evaluate("./training_data/income/", preferences, project_name="IncomeMultiplicityEvaluation_centralized", run_name="centralized")
