import concurrent.futures
import io
import timeit
from logging import INFO, WARNING
import pickle as pkl

import numpy


from ClientManager.client_manager import SimpleClientManager
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar, parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import FedAvg, Strategy


from Strategy.custom_evaluation import CustomEvaluation
from Utils.preferences import Preferences

FitResultsAndFailures = tuple[
    list[tuple[ClientProxy, FitRes]],
    list[tuple[ClientProxy, FitRes] | BaseException],
]
EvaluateResultsAndFailures = tuple[
    list[tuple[ClientProxy, EvaluateRes]],
    list[tuple[ClientProxy, EvaluateRes] | BaseException],
]
ReconnectResultsAndFailures = tuple[
    list[tuple[ClientProxy, DisconnectRes]],
    list[tuple[ClientProxy, DisconnectRes] | BaseException],
]


class EvalServer:
    """Flower server only for evaluation."""

    def __init__(
        self,
        *,
        client_manager: SimpleClientManager,
        preferences: Preferences,
        strategy: Strategy | None = None,
    ) -> None:
        self._client_manager: SimpleClientManager = client_manager
        self.parameters:dict[str, Parameters]
        self.strategy=strategy
        self.max_workers: int | None = None
        self.preferences = preferences

    def set_max_workers(self, max_workers: int | None) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> SimpleClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
        """Run federated averaging for a number of rounds."""
        history = History()
        start_time = timeit.default_timer()
        # Initialize parameters
        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Starting evaluation for Rashomon set")
        res = self.test_round(0, timeout=timeout)
        if res is not None:
            log(
                INFO,
                "accuracy of models:  %s",
                res[0],
            )

        else:
            log(INFO, "Rashomon set returned no results (`None`)")

        log(INFO, "Starting multiplicity evaluation for each Rashomon set")
        res = self.evaluate_round(0, timeout=timeout)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )

        else:
            log(INFO, "Multiplicity evaluation returned no results (`None`)")





        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed

    def test_round(
            self,
            server_round: int,
            timeout: float | None,
    ) -> tuple[ dict[str, Scalar], EvaluateResultsAndFailures] | None:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy

        para=(list(self.parameters.values())[0])
        client_instructions = self.strategy.configure_test(
            server_round=server_round,
            parameters=para,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "configure_rashomon_test: no clients selected, skipping rashomon test")
            return None
        log(
            INFO,
            "configure_rashomon_test: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(phase="test"),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_rashomon_test: received %s results and %s failures",
            len(results),
            len(failures),
        )
        log(INFO, "Failures: %s", failures)

        # Aggregate the evaluation results
        aggregated_result:dict[str, Scalar]= self.strategy.aggregate_test(server_round, results, failures)


        return aggregated_result, (results, failures)

    def evaluate_round(
        self,
        server_round: int,
        timeout: float | None,
    ) -> tuple[float | None, dict[str, Scalar], EvaluateResultsAndFailures] | None:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        para = (list(self.parameters.values())[0])
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=para,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "configure_evaluate_rashomon_set: no clients selected, skipping evaluation")
            return None
        log(
            INFO,
            "configure_evaluate_rashomon_set: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(phase="validation"),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_multiplicity_metrics: received %s results and %s failures",
            len(results),
            len(failures),
        )
        log(INFO, "Failures: %s", failures)

        # Aggregate the evaluation results
        aggregated_result: dict[str, Scalar]= self.strategy.aggregate_evaluate(server_round, results, failures)


        return aggregated_result, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: float | None,
    ) -> tuple[Parameters | None, dict[str, Scalar], FitResultsAndFailures] | None:
        """Perform a single round of federated averaging."""
        pass

    def disconnect_all_clients(self, timeout: float | None) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, server_round: int, timeout: float | None) -> dict[str,Parameters]:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: dict[str,Parameters] | None = self.strategy.initialize_parameters(client_manager=self._client_manager)
        if parameters is not None:
            log(INFO, "Using initial global parameters provided by strategy")
            return parameters
        else:
            log(
                WARNING,
                "Provide parameters from dictionary",
            )



def reconnect_clients(
    client_instructions: list[tuple[ClientProxy, ReconnectIns]],
    max_workers: int | None,
    timeout: float | None,
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout) for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: list[tuple[ClientProxy, DisconnectRes]] = []
    failures: list[tuple[ClientProxy, DisconnectRes] | BaseException] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: float | None,
) -> tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
        group_id=None,
    )
    return client, disconnect


def fit_clients(
    client_instructions: list[tuple[ClientProxy, FitIns]],
    max_workers: int | None,
    timeout: float | None,
    group_id: int,
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    pass


def fit_client(client: ClientProxy, ins: FitIns, timeout: float | None, group_id: int) -> tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    pass


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: list[tuple[ClientProxy, FitRes]],
    failures: list[tuple[ClientProxy, FitRes] | BaseException],
) -> None:
    """Convert finished future into either a result or a failure."""
    pass


def evaluate_clients(
    client_instructions: list[tuple[ClientProxy, EvaluateIns]],
    max_workers: int | None,
    timeout: float | None,
    group_id: int,
    # phase: str,
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout, group_id)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: list[tuple[ClientProxy, EvaluateRes]] = []
    failures: list[tuple[ClientProxy, EvaluateRes] | BaseException] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(future=future, results=results, failures=failures)

    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: float | None,
    group_id: int,
    # phase: str,
) -> tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout, group_id=group_id)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: list[tuple[ClientProxy, EvaluateRes]],
    failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def run_fl(
    server: EvalServer,
    config: ServerConfig,
) -> History:
    """Train a model on the given server and return the History object."""
    log(INFO, "right server starting")
    hist, elapsed_time = server.evaluate_round(server_round=1, timeout=config.round_timeout)

    log(INFO, "")
    log(INFO, "[SUMMARY]")
    log(INFO, "Run finished %s round(s) in %.2fs", config.num_rounds, elapsed_time)
    for line in io.StringIO(str(hist)):
        log(INFO, "\t%s", line.strip("\n"))
    log(INFO, "")

    # Graceful shutdown
    server.disconnect_all_clients(timeout=config.round_timeout)

    return hist
