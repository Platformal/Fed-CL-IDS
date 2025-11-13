from flwr.common import ArrayRecord, ConfigRecord, MetricRecord
from flwr.common import RecordDict, Message, MessageType, log
from flwr.serverapp.strategy.strategy_utils import sample_nodes
from flwr.serverapp.strategy import FedAvg, Result
from flwr.server import Grid
from logging import INFO

from collections.abc import Iterable
from typing import Callable, Optional
from torch import Tensor
import torch
import time
import io
import json

class UAVIDSFedAvg(FedAvg):
    def __init__(self, fraction_train: float, fraction_eval: float) -> None:
        super().__init__(fraction_train, fraction_eval)
        self.train_node_ids: list[int] = []
        self.evaluate_node_ids: list[int] = []
        self.all_node_ids: list[int] = []
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
    
    # Changed sample_nodes function only to initialize
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, 
        config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        """Configure the next round of federated training"""
        if self.fraction_train == 0.0:
            return []
        if not self.train_node_ids:
            num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
            sample_size = max(num_nodes, self.min_train_nodes)
            node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
            self.all_node_ids = num_total
            self.train_node_ids = node_ids
        log(
            INFO,
            f"configure_train: Sampled {len(self.train_node_ids)} nodes "
            f"(out of {len(self.all_node_ids)})"
        )
        config['server-round'] = server_round

        record = RecordDict({
            self.arrayrecord_key: arrays, 
            self.configrecord_key: config
        })
        messages = self._construct_messages(record, self.train_node_ids, MessageType.TRAIN)
        # Assign flows to messages
        flows: list[list[int]] = json.loads(config['flows'])
        for i, message in enumerate(messages):
            client_content = message.content.copy()
            client_content['config'] = client_content['config'].copy()
            client_content['config']['flows'] = flows[i]
            message.content = client_content
        return messages

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation"""
        if self.fraction_evaluate == 0.0:
            return []

        if not self.evaluate_node_ids:
            num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_evaluate)
            sample_size = max(num_nodes, self.min_evaluate_nodes)
            node_ids, _ = sample_nodes(grid, self.min_available_nodes, sample_size)
            self.evaluate_node_ids = node_ids
        log(
            INFO,
            f"configure_evaluate: Sampled {len(self.evaluate_node_ids)} nodes "
            f"(out of {len(self.all_node_ids)})",
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict({
            self.arrayrecord_key: arrays,
            self.configrecord_key: config
        })
        messages = self._construct_messages(record, self.evaluate_node_ids, MessageType.EVALUATE)
        flows: list[list[int]] = json.loads(config['flows'])
        for i, message in enumerate(messages):
            client_content = message.content.copy()
            client_content['config'] = client_content['config'].copy()
            client_content['config']['flows'] = flows[i]
            message.content = client_content
        return messages

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        current_day: int,
        num_rounds: int,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None
    ) -> Result:
        """Execute the federated learning strategy.

        Runs the complete federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        num_rounds : int (default: 3)
            Number of federated learning rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Optional function for centralized evaluation of the global model. Takes
            server round number and array record, returns a MetricRecord or None. If
            provided, will be called before the first round and after each round.
            Defaults to None.

        Returns
        -------
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
        """
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        current_array = initial_arrays
        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[DAY %s | ROUND %s/%s]", current_day, current_round, num_rounds)
            
            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    current_array,
                    train_config,
                    grid,
                ),
                timeout=timeout
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                current_array = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies: Iterable[Message] = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    current_array,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate (Can add GPU if need be)
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )
            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

        # -----------------------------------------------------------------
        # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
        # -----------------------------------------------------------------

        # Centralized evaluation
        if evaluate_fn:
            log(INFO, "Global evaluation")
            res = evaluate_fn(current_round, current_array)
            log(INFO, "\t└──> MetricRecord: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[current_round] = res

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
    
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None
        
        reply_contents = [msg.content for msg in valid_replies]
        arrays: list[dict[str, Tensor]] = [
            content[self.arrayrecord_key].to_torch_state_dict() 
            for content in reply_contents
        ]

        aggregated_model: dict[str, Tensor] = {}
        with torch.no_grad():
            # Parameter key example: 'network.9.weight', 'network.9.bias', ...
            for parameter_key in arrays[0].keys():
                tensors_of_key: Tensor = torch.stack([
                    torch.as_tensor(array[parameter_key]).to(self.device)
                    for array in arrays
                ], dim=0)
                averaged_tensor = torch.mean(tensors_of_key, dim=0)
                aggregated_model[parameter_key] = averaged_tensor.cpu()
        array_record = ArrayRecord(aggregated_model)

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(
            reply_contents,
            self.weighted_by_key,
        )
        return array_record, metrics
    
def str_config(config: ConfigRecord) -> str:
    """Ensures start log does not print all elements in flows"""
    all_config_str: list[str] = []
    for key, value in config.items():
        try:
            value = json.loads(value)
        except:
            value = value
        if isinstance(value, bytes):
            string = f"'{key}': '<bytes>'"
        elif isinstance(value, list):
            string = f"'{key}': length = {len(value)}"
        else:
            string = f"'{key}': '{value}'"
        all_config_str.append(string)
    content = ", ".join(all_config_str)
    return f"{{{content}}}"

def log_strategy_start_info(
    num_rounds: int,
    arrays: ArrayRecord,
    train_config: Optional[ConfigRecord],
    evaluate_config: Optional[ConfigRecord],
) -> None:
    """Log information about the strategy start."""
    log(INFO, "\t├── Number of rounds: %d", num_rounds)
    log(
        INFO,
        "\t├── ArrayRecord (%.2f MB)",
        sum(len(array.data) for array in arrays.values()) / (1024**2),
    )
    log(
        INFO,
        "\t├── ConfigRecord (train): %s",
        str_config(train_config) if train_config else "(empty!)",
    )
    log(
        INFO,
        "\t├── ConfigRecord (evaluate): %s",
        str_config(evaluate_config) if evaluate_config else "(empty!)",
    )
