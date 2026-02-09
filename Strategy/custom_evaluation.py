# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""
import json
import os
import shutil
from collections.abc import Callable
from logging import WARNING
from typing import Any

import numpy
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy

from ClientManager.client_manager import SimpleClientManager
from Utils.preferences import Preferences

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class CustomEvaluation(Strategy):
    """
    Federated Averaging evaluation on multiple models implementation.

    Based on McMahan et al. (2016). Supports client sampling for fit/evaluate/test, weighted aggregation of parameters and metrics, initial parameters, and custom config functions.

    Args:
        fraction_fit (float, optional): Fraction of clients for training. Defaults to 1.0.
        fraction_evaluate (float, optional): Fraction of clients for evaluation. Defaults to 1.0.
        min_fit_clients (int, optional): Minimum clients for training. Defaults to 2.
        min_evaluate_clients (int, optional): Minimum clients for evaluation. Defaults to 2.
        min_available_clients (int, optional): Minimum total clients. Defaults to 2.
        preferences (Preferences): User preferences containing FL settings.
        evaluate_fn (Callable, optional): Central evaluation function. Defaults to None.
        on_fit_config_fn (Callable, optional): Config for training round. Defaults to None.
        on_evaluate_config_fn (Callable, optional): Config for evaluation round. Defaults to None.
        accept_failures (bool, optional): Accept rounds with failures. Defaults to True.
        initial_parameters (Parameters, optional): Initial global parameters. Defaults to None.
        fit_metrics_aggregation_fn (MetricsAggregationFn, optional): Aggregation for fit metrics. Defaults to None.
        evaluate_metrics_aggregation_fn (MetricsAggregationFn, optional): Aggregation for evaluate metrics. Defaults to None.
        test_metrics_aggregation_fn (MetricsAggregationFn, optional): Aggregation for test metrics. Defaults to None.
        inplace (bool, optional): In-place aggregation. Defaults to True.
        wandb_run (Any, optional): WandB run for logging. Defaults to None.

    Returns:
        None
    """

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> list[
        tuple[ClientProxy, FitIns]]:
        pass

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]],
                      failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[
        Parameters | None, dict[str, Scalar]]:
        pass

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_evaluate: float = 1.0,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        preferences: Preferences,
        evaluate_fn: Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: dict[str,Parameters]|None=None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        inplace: bool = True,
        wandb_run: Any = None,
        epsilons: list[float] = [0.0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.024, 0.028, 0.032, 0.036, 0.04],

    ) -> None:
        """
        Initializes the FedAvg strategy with sampling and aggregation parameters.

        Validates min_available_clients; sets attributes for client sampling, config functions, aggregation, etc.

        Args:
            fraction_fit (float, optional): Fraction of clients for fit. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients for evaluate. Defaults to 1.0.
            min_fit_clients (int, optional): Min clients for fit. Defaults to 2.
            min_evaluate_clients (int, optional): Min clients for evaluate. Defaults to 2.
            min_available_clients (int, optional): Min total clients. Defaults to 2.
            preferences (Preferences): FL preferences.
            evaluate_fn (Callable, optional): Central eval function. Defaults to None.
            on_fit_config_fn (Callable, optional): Fit config fn. Defaults to None.
            on_evaluate_config_fn (Callable, optional): Evaluate config fn. Defaults to None.
            accept_failures (bool, optional): Accept failures. Defaults to True.
            initial_parameters (Parameters, optional): Initial params. Defaults to None.
            fit_metrics_aggregation_fn (MetricsAggregationFn, optional): Fit metrics agg fn. Defaults to None.
            evaluate_metrics_aggregation_fn (MetricsAggregationFn, optional): Evaluate metrics agg fn. Defaults to None.
            test_metrics_aggregation_fn (MetricsAggregationFn, optional): Test metrics agg fn. Defaults to None.
            inplace (bool, optional): In-place agg. Defaults to True.
            wandb_run (Any, optional): WandB run. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If min_available_clients too low (logged as warning).
        """
        super().__init__()

        self.baseline_accuracy = preferences.baseline_accuracy
        self.epsilons = epsilons
        if min_evaluate_clients > min_available_clients:
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_evaluate = fraction_evaluate
        self.preferences = preferences

        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn

        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters

        self.test_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace

        self.wandb_run = wandb_run
        self.fed_dir = preferences.fed_dir

    def __repr__(self) -> str:
        """
        Returns a string representation of the FedAvg strategy.

        Args:
            None

        Returns:
            str: Representation including accept_failures flag.
        """
        rep = "CustomEvaluation"
        return rep


    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """
        Computes the number of clients to sample for evaluation and the min available required.

        Uses fraction_evaluate, ensures at least min_evaluate_clients.

        Args:
            num_available_clients (int): Total available clients.

        Returns:
            tuple[int, int]: (sample_size, min_available_clients)
        """
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(self, client_manager: SimpleClientManager) -> dict[str, Parameters]:
        """
        Initializes global model parameters using provided initial_parameters or strategy default.

        Clears initial_parameters from memory after use.

        Args:
            client_manager (SimpleClientManager): Client manager (unused in this implementation).

        Returns:
            Parameters | None: Initial global parameters.
        """
        initial_parameters = self.initial_parameters
        #self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(self, server_round: int, parameters: Parameters) -> tuple[float, dict[str, Scalar]] | None:
        """
        Performs central evaluation using the provided evaluate_fn.

        Converts parameters to NDArrays, calls evaluate_fn if available.

        Args:
            server_round (int): Current server round.
            parameters (Parameters): Global model parameters.

        Returns:
            tuple[float, dict[str, Scalar]] | None: (loss, metrics) or None if no evaluate_fn.
        """
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics


    def configure_test(
        self, server_round: int, parameters: Parameters, client_manager: SimpleClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """
        Configures the test instructions for the next test round.

        Skips if fraction_evaluate is 0; generates config if on_evaluate_config_fn provided, samples clients using num_evaluation_clients for test phase.

        Args:
            server_round (int): Current server round.
            parameters (Parameters): Global parameters to send to clients.
            client_manager (SimpleClientManager): Manager to sample clients from.

        Returns:
            list[tuple[ClientProxy, EvaluateIns]]: List of (client, test instructions) pairs.

        Raises:
            ValueError: If no clients can be sampled.
        """
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {"path": self.fed_dir, "phase": "Rashomon"}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available(phase="test"))
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            phase="test",
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters:  Parameters, client_manager: SimpleClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """
        Configures the evaluate instructions for the next validation round.

        Skips if fraction_evaluate is 0; generates config if on_evaluate_config_fn provided, samples clients using num_evaluation_clients for validation phase.

        Args:
            server_round (int): Current server round.
            parameters (Parameters): Global parameters to send to clients.
            client_manager (SimpleClientManager): Manager to sample clients from.

        Returns:
            list[tuple[ClientProxy, EvaluateIns]]: List of (client, evaluate instructions) pairs.

        Raises:
            ValueError: If no clients can be sampled.
        """
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {"path": self.fed_dir, "phase": "Multiplicity"}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available(phase="validation"))
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            phase="validation",
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_test(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, EvaluateRes]],
            failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """
        Aggregates test results: weighted average loss, custom metrics if fn provided.

        Skips if no results or unacceptable failures.

        Args:
            server_round (int): Current server round.
            results (list[tuple[ClientProxy, EvaluateRes]]): Successful test results.
            failures (list[tuple[ClientProxy, EvaluateRes] | BaseException]): Failures.

        Returns:
            tuple[float | None, dict[str, Scalar]]: Aggregated loss and metrics.

        Raises:
            ValueError: If aggregation fails.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}


        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        metrics = [(res.num_examples, res.metrics, cid.cid) for cid, res in results]


        client_agg_metrics = {

            "FL Round": server_round,
        }
        for n_examples, node_metrics, node_name in metrics:
            for key, value in node_metrics.items():
                client_agg_metrics[f"Rashomon Set Test Node {node_name} - Model {key}"] = value
        if self.wandb_run:
            self.wandb_run.log(client_agg_metrics)

        total_examples = sum([n_examples for n_examples, _, _ in metrics])
        agg_metrics = {}

        for key,_ in self.initial_parameters.items():
            if self.preferences.threshold is not None:
                agg_metrics[f"{key}_accuracy"] = [abs(metric[f"{key}_acc"]-self.baseline_accuracy) for _,metric,_ in metrics]
                n = len(agg_metrics[f"{key}_accuracy"])
                k = int(np.ceil(self.preferences.threshold * n))
                # percentile that leaves k elements greater → 100 - (k/n)*100
                agg_metrics[f"{key}_accuracy"]=np.percentile(agg_metrics[f"{key}_accuracy"], (k / n) * 100, interpolation='higher')
                #agg_metrics[f"{key}_accuracy"] = [metric[f"{key}"]-self.baseline_accuracy for _,metric,_ metrics]
            else:
                agg_metrics[f"{key}_accuracy"] = sum([n_examples * metric[f"{key}_acc"] for n_examples, metric, _ in metrics]) / total_examples

            dp_arrays = [
                np.array(json.loads(metric[f"{key}_dp"]))
                for _, metric, _ in metrics
            ]
            n = len(dp_arrays[0])
            assert all(len(arr) == n for arr in dp_arrays), "Inconsistent demographic parity array lengths"
            demo_par = np.sum(dp_arrays, axis=0)
            demo_par = [demo_par[0]/demo_par[2], demo_par[1]/demo_par[3]]

            # Calculate demographic parity
            agg_metrics[f"{key}_demographic_parity"] = np.max(demo_par) - np.min(demo_par)
        agg_metrics["FL Round"] = server_round



        #if self.wandb_run:
         #   self.wandb_run.log(agg_metrics)

            # Create base directory if it doesn't exist
        os.makedirs(f"Rashomon_Set_{self.preferences.dataset_name}", exist_ok=True)
        #remove the old Rashomon set directories
        for item in os.listdir(f"Rashomon_Set_{self.preferences.dataset_name}/"):
            item_path = os.path.join(f"Rashomon_Set_{self.preferences.dataset_name}/", item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # remove directory

        # Create epsilon directories
        epsilon_dirs = {}
        epsilon_counts = {}
        epsilon_counts[None] = 0
        for epsilon in sorted(self.epsilons):
            dir_name = f"Rashomon_Set_{self.preferences.dataset_name}/epsilon_{epsilon}"
            os.makedirs(dir_name, exist_ok=True)
            epsilon_dirs[epsilon] = dir_name
            epsilon_counts[epsilon] = 0

        # Process each model
        processed_models = set()  # To ensure each model is only sorted once

        for key, accuracy in agg_metrics.items():
            if not key.endswith("_accuracy"):
                continue

            model_name = key.replace("_accuracy", "")

            # Skip if we've already processed this model
            if model_name in processed_models:
                continue

            # Calculate difference from baseline
            if self.preferences.threshold is not None:
                diff = accuracy
            else:
                diff = abs(accuracy - self.baseline_accuracy)

            # Find appropriate epsilon range
            assigned_epsilon = None
            for epsilon in sorted(self.epsilons):
                if diff <= epsilon:
                    assigned_epsilon = epsilon
                    break

            # If no epsilon range matches, put in a "no_match" directory
            if assigned_epsilon is None:
                dir_name = f"Rashomon_Set_{self.preferences.dataset_name}/no_match"
                os.makedirs(dir_name, exist_ok=True)
                target_dir = dir_name
                epsilon_counts[None] += 1


            else:
                target_dir = epsilon_dirs[assigned_epsilon]
                epsilon_counts[assigned_epsilon] += 1

            try:

                model_file_path = f"{target_dir}/{model_name}.npz"
                arrays = parameters_to_ndarrays(self.initial_parameters[model_name])
                numpy.savez(model_file_path, *arrays)


                print(f"Saved {model_name} to {target_dir}")

            except Exception as e:
                print(f"Error saving {model_name}: {e}")

            processed_models.add(model_name)

        agg_metrics["rashomonratio"] = [float(epsilon_counts[epsilon]) for epsilon in sorted(self.epsilons)]
        print(agg_metrics["rashomonratio"])
        if self.wandb_run:
            self.wandb_run.log(agg_metrics)

        return agg_metrics

        return loss_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """
        Aggregates evaluate results: weighted average loss, custom metrics if fn provided.

        Skips if no results or unacceptable failures.

        Args:
            server_round (int): Current server round.
            results (list[tuple[ClientProxy, EvaluateRes]]): Successful evaluate results.
            failures (list[tuple[ClientProxy, EvaluateRes] | BaseException]): Failures.

        Returns:
            tuple[float | None, dict[str, Scalar]]: Aggregated loss and metrics.

        Raises:
            ValueError: If aggregation fails.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        metrics = [(res.num_examples, res.metrics, cid.cid) for cid, res in results]

        total_examples = sum([n_examples for n_examples, _, _ in metrics])
        agg_metrics = {}

        amb = [n_examples*np.array(json.loads(metric["amb"])) for n_examples, metric, _ in metrics]
        agg_metrics["amb"] = np.sum(amb,axis=0) / total_examples
        #print("ambiguity", amb,agg_metrics["amb"])

        disc = [n_examples/total_examples * np.array(json.loads(metric["disc"])) for n_examples, metric, _ in metrics]
        # print(disc)
        agg_metrics["disc"] = np.max(disc, axis=0)
        #print("discrepancy", disc,agg_metrics["disc"])

        disa_hat =np.cumsum(np.sum([json.loads(metric["disa_hat_difpriv"]) for n_examples, metric, _ in metrics], axis=0), axis=1)
        disa_hat_cum = [entry/entry[-1] for entry in disa_hat]

        agg_metrics["disa_hat_90_difpriv"] = [float(np.searchsorted(entry, 0.9, "left")*0.001) for entry in disa_hat_cum]
        agg_metrics["disa_hat_50_difpriv"] =[float(np.searchsorted(entry, 0.5)*0.001)for entry in disa_hat_cum]
        for entry in disa_hat_cum:
            entry[np.isnan(entry)] = 0
            #disa_hat_cum[entry] = entry
        #print(disa_hat, disa_hat_cum)
        agg_metrics["disa_hat_difpriv"] = disa_hat_cum
        #print(agg_metrics[f"disa_hat_90_difpriv"], agg_metrics[f"disa_hat_50_difpriv"])

        disa_hat = [json.loads(metric["disa_hat"]) for n_examples, metric, _ in metrics]
        disa_hat = [[e for entry in disa_hat for e in entry[i]] for i in range(len(disa_hat[0]))]
        agg_metrics["disa_hat_90"] = np.quantile(disa_hat, 0.9, axis=1)
        agg_metrics["disa_hat_50"] =np.quantile(disa_hat, 0.5, axis=1)
            #disa_hat_cum[entry] = entry
        #print(disa_hat, disa_hat_cum)
        agg_metrics["disa_hat"] = disa_hat

        vpr = np.cumsum(np.sum([json.loads(metric["vpr_difpriv"]) for n_examples, metric, _ in metrics], axis=0), axis=1)
        vpr_cum = [entry/entry[-1] for entry in vpr]

        agg_metrics["vpr_90_difpriv"] =  [float(np.searchsorted(entry, 0.9)*0.001) for entry in vpr_cum]
        agg_metrics["vpr_50_difpriv"] = [float(np.searchsorted(entry, 0.5) *0.001)for entry in vpr_cum]
        for entry in vpr_cum:
            entry[np.isnan(entry)] = 0
        #print(vpr, vpr_cum)
        agg_metrics["vpr_difpriv"] = vpr_cum
        #print(agg_metrics[f"vpr_90_difpr"], agg_metrics[f"vpr_50"])

        vpr = [json.loads(metric["vpr"]) for n_examples, metric, _ in metrics]
        vpr = [[e for entry in vpr for e in entry[i]] for i in range(len(vpr[0]))]
        agg_metrics["vpr_90"] =np.quantile(vpr, 0.9, axis=1)
        print (vpr[2][:10])
        agg_metrics["vpr_50"] =np.quantile(vpr, 0.5, axis=1)
        agg_metrics["vpr"] = vpr

        score_var = np.cumsum(np.sum([json.loads(metric["score_var_difpriv"]) for n_examples, metric, _ in metrics], axis=0), axis=1)
        score_var_cum = [entry/entry[-1] for entry in score_var]
        #print(score_var, score_var_cum)

        agg_metrics["score_var_90_difpriv"] =  [float(np.searchsorted(entry, 0.9) *0.001)for entry in score_var_cum]
        agg_metrics["score_var_50_difpriv"] = [float(np.searchsorted(entry, 0.5)*0.001) for entry in score_var_cum]
        for entry in score_var_cum:
            entry[np.isnan(entry)] = 0
        agg_metrics["score_var_difpriv"] = score_var_cum
        #print(agg_metrics[f"score_var_90"], agg_metrics[f"score_var_50"])

        score_var = [json.loads(metric["score_var"]) for n_examples, metric, _ in metrics]
        score_var = [[e for entry in score_var for e in entry[i]] for i in range(len(score_var[0]))]
        agg_metrics["score_var_90"] = np.quantile(score_var, 0.9, axis=1)
        agg_metrics["score_var_50"] =np.quantile(score_var, 0.5, axis=1)
        agg_metrics["score_var"] = score_var
        # print(score_var, score_var_cum)

        if self.wandb_run:
            self.wandb_run.log(agg_metrics)

        #print(agg_metrics)
        agg_metrics["FL Round"] = server_round



        return agg_metrics
