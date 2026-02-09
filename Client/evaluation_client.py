import json
import os
from typing import Any, List

import numpy
import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, Parameters
from Models.regression_model import RegressionModel
from Models.simple_model import SimpleModel
from Models.utils import get_model
from torch import nn
from torch.utils.data import DataLoader
from Utils.multiplicity_evaluation import *

# from Training.training import test, train
from Utils.preferences import Preferences
from Utils.utils import get_optimizer, get_params, set_params


class FlowerClient(NumPyClient):
    def __init__(
        self, trainloader: DataLoader, valloader: DataLoader, preferences: Preferences, partition_id: int
    ) -> None:
        """
        Initializes a Flower client instance for federated learning.

        Sets up data loaders, device, preferences, and model (SimpleModel for classification or RegressionModel for regression).

        Args:
            trainloader (DataLoader): DataLoader for training data.
            valloader (DataLoader): DataLoader for validation data.
            preferences (Preferences): Configuration preferences for the FL setup.
            partition_id (int): Unique identifier for this client's data partition.

        Returns:
            None
        """
        super().__init__()

        self.partition_id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.preferences = preferences
        trained_model = get_model(dataset=preferences.dataset_name)
        if self.preferences.task == "classification":
            self.model = SimpleModel(
                model=trained_model,
                optimizer=get_optimizer(trained_model, self.preferences),
                criterion=nn.CrossEntropyLoss(),
                device=self.device,
            )
        elif self.preferences.task == "regression":
            self.model = RegressionModel(
                model=trained_model,
                optimizer=get_optimizer(trained_model, self.preferences),
                criterion=nn.MSELoss(),
                device=self.device,
            )
        else:
            error = f"Unknown task type: {self.preferences.task}"
            raise ValueError(error)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Any]]:
        """
        Performs local training on the client's data using parameters received from the server.

        Updates the local model parameters over the specified number of epochs and returns updated parameters along with training metrics.

        Args:
            parameters (NDArrays): Model parameters from the server.
            config (dict[str, Scalar]): Configuration dictionary from the server.

        Returns:
            tuple[NDArrays, int, dict[str, Any]]: Updated model parameters, number of training examples, and training result dictionary (e.g., containing loss and accuracy).

        Raises:
            RuntimeError: If training fails due to device or model issues.
        """
        # copy parameters sent by the server into client's local model
        set_params(self.model.model, parameters)

        # do local training (call same function as centralised setting)
        for _ in range(self.preferences.num_epochs):
            result_dict = self.model.train(trainloader=self.trainloader, epochs=self.preferences.num_epochs)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.model.model), len(self.trainloader), result_dict

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Any]]:
        """
        Evaluates the model using parameters received from the server on the client's validation set.

        Computes loss and other metrics (e.g., accuracy for classification, rmse/mae for regression).

        Args:
            parameters (NDArrays): Model parameters from the server.
            config (dict[str, Scalar]): Configuration dictionary from the server.

        Returns:
            tuple[float, int, dict[str, Any]]: Evaluation loss, number of validation examples, and evaluation result dictionary with metrics.

        Raises:
            RuntimeError: If evaluation fails due to device or model issues.
        """

        if config["phase"]=="Rashomon":
            path = config["path"]
            global_model_init = {}
            result_dict = {}
            for dir in os.listdir(f"{path}"):
                if os.path.isdir(os.path.join(f"{path}", dir)):
                    #ndarrays = numpy.load(f"{path}/{dir}/model.npz")
                    # Convert model parameters to flwr.common.Parameters
                    #global_model_init[dir] = ([ndarrays["arr_0"]])

                    global_model_init[dir] = load_parameters_from_file(f"{path}/{dir}/model.npz")
            for key, value in global_model_init.items():
                set_params(self.model.model, value)
                evaluation = self.model.evaluate(testloader=self.valloader)
                result_dict[f"{key}_acc"] = evaluation["accuracy"]
                result_dict[f"{key}_dp_local"] = evaluation["local_dp"]
                result_dict[f"{key}_dp"] = json.dumps(evaluation["dp"])

            return 0.0, len(self.valloader), result_dict
        else:

            # Define base directory where models are stored
            base_directory = f"Rashomon_Set_{self.preferences.dataset_name}"
            base_model_params= load_parameters_from_file(self.preferences.baseline_model+ "/model.npz")
            set_params(self.model.model, base_model_params)
            self.model.model.eval()
            base_pred_y =[]
            for sample, _, label in self.valloader:
                images, labels = sample.to(self.device), label.to(self.device)
                outputs = self.model.model(images)
                base_pred_y.append(torch.max(outputs.data, 1)[1])

            base_pred_y = [x for xs in base_pred_y for x in xs]
            # Load the sorted models from directories
            #print(base_pred_y)


            # Load models in epsilon order
            sorted_models = load_models_sorted_by_epsilon(base_directory)
            len_eps_list = len(sorted_models.items())

            all_sampling_test_scores = []

            i=0

            # Process models in epsilon order
            ntest = len(base_pred_y)
            vpr_difpriv = np.zeros((len_eps_list, 1000))
            score_var_difpriv = np.zeros((len_eps_list, 1000))
            #rc_difpriv = np.zeros((len_eps_list, 1000))
            vpr = np.zeros((len_eps_list, ntest))
            score_var = np.zeros((len_eps_list, ntest))
            #rc = np.zeros((len_eps_list, ntest))
            amb = np.zeros((len_eps_list,))
            disc = np.zeros((len_eps_list,))
            disa_hat_difpriv = np.zeros((len_eps_list, 1000))
            disa_hat = np.zeros((len_eps_list, ntest))
            for epsilon, model_paths in sorted_models.items():
                for model_path in model_paths:
                    # Load model parameters
                    sampling_test_scores=[]
                    model_params = load_parameters_from_file(model_path)
                    set_params(self.model.model, model_params)
                    self.model.model.eval()
                    for sample, _, label in self.valloader:
                        images, labels = sample.to(self.device), label.to(self.device)
                        outputs = self.model.model(images)
                        sampling_test_scores.append(outputs.data)
                    all_sampling_test_scores.append([x for xs in sampling_test_scores for x in xs])

                if not all_sampling_test_scores:
                    i += 1
                    continue

                scores = score_of_y_multi_model(all_sampling_test_scores, base_pred_y)
                #scores = all_sampling_test_scores[0]
                #for j in range(1, len(all_sampling_test_scores)):
                #    scores = np.concatenate((scores, list(all_sampling_test_scores[j])),
                 #               axis=0)

                vpr_difpriv[i,:],vpr[i, :] = viable_prediction_range(scores)
                score_var_difpriv[i, :], score_var[i,:] = score_variance(scores)
                #rc[i, :] = rashomon_capacity(all_sampling_test_scores)
                #print(rc)


                decisions =  [[torch.max(subsubarray,0)[1] for subsubarray in subarray] for subarray in all_sampling_test_scores ]
                amb[i] = ambiguity(decisions, base_pred_y)
                disc[i] = discrepancy(decisions, base_pred_y)
                disa_hat_difpriv[i, :], disa_hat[i,:] = disagreement_hat(decisions)


                i+=1

            savename =  f'{self.partition_id}_results.npz'
            np.savez_compressed(config["path"] + savename,
                                all_sampling_test_scores=all_sampling_test_scores,
                                eps_list=list(sorted_models.keys()),
                                vpr=vpr,
                                score_var=score_var,
                                amb=amb,
                                disc=disc,
                                disa_hat=disa_hat,

                                )
            #print(vpr, score_var, rc, amb, disc, disa_hat)
            #print(amb)
            vpr_string = json.dumps(vpr.tolist())
            score_var_string =  json.dumps(score_var.tolist())
            vpr_string_difpriv = json.dumps(vpr_difpriv.tolist())
            score_var_string_difpriv = json.dumps(score_var_difpriv.tolist())
            #rc_string = json.dumps(rc.tolist())
            amb_string = json.dumps(amb.tolist())
            disc_string =  json.dumps(disc.tolist())
            disa_hat_string = json.dumps(disa_hat.tolist())
            disa_hat_string_difpriv = json.dumps(disa_hat_difpriv.tolist())
            return 0.0, len(self.valloader), {
                                "vpr":vpr_string,
                                "score_var":score_var_string,
                                "vpr_difpriv": vpr_string_difpriv,
                                "score_var_difpriv": score_var_string_difpriv,
                                #"rc":rc_string,
                                "amb":amb_string,
                                "disc":disc_string,
                                "disa_hat":disa_hat_string,
                                "disa_hat_difpriv": disa_hat_string_difpriv
                                }




def load_models_sorted_by_epsilon(base_directory: str) -> dict:
                """Load models sorted by epsilon ranges"""
                epsilon_dirs = {}

                # Get all epsilon directories
                for item in os.listdir(base_directory):
                    item_path = os.path.join(base_directory, item)
                    if os.path.isdir(item_path) and item.startswith('epsilon_'):
                        epsilon_str = item.replace('epsilon_', '')
                        try:
                            epsilon_val = float(epsilon_str)
                            epsilon_dirs[epsilon_val] = item_path
                        except ValueError:
                            continue

                # Sort by epsilon values
                sorted_epsilons = sorted(epsilon_dirs.keys())

                # Load models in order
                loaded_models = {}
                for epsilon in sorted_epsilons:
                    models_in_dir = []
                    dir_path = epsilon_dirs[epsilon]

                    # Look for model files (adjust extension as needed)
                    for file in os.listdir(dir_path):
                        if file.endswith('.npz'):  # Adjust extension as needed
                            model_path = os.path.join(dir_path, file)
                            models_in_dir.append(model_path)

                    loaded_models[epsilon] = models_in_dir

                return loaded_models

            # Function to load parameters from saved file
# def load_parameters_from_file(file_path: str) -> NDArrays:
#     """Load Parameters from file"""
#     # Adjust this based on how you saved your parameters
#     with open(file_path, 'rb') as f:
#         ndarrays = numpy.load(file_path)
#         # Convert model parameters to flwr.common.Parameters
#         parameters = [ndarrays["arr_0"]]
#     return parameters


def load_parameters_from_file(file_path: str) -> NDArrays:
    """Load all parameters from a .npz file generically."""
    
    # numpy.load acts as a context manager, so we don't need 'open()'
    with np.load(file_path) as data:
        # data.files is a list of all keys in the file (e.g., ['arr_0', 'arr_1', ...])
        # We extract every array associated with these keys.
        parameters = [data[key] for key in data.files]
        
    return parameters