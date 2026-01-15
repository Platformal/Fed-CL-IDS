from typing import Callable, Optional, Iterable, Container, cast
from collections import OrderedDict
from logging import INFO
import json
import time
import io

from flwr.common import (
    ArrayRecord, ConfigRecord, MetricRecord,
    RecordDict, Message, MessageType, log
)
from flwr.serverapp.strategy.strategy_utils import sample_nodes
from flwr.serverapp.strategy import FedAvg, Result
from flwr.app import Context
from flwr.server import Grid

from torch import Tensor
import torch

class FedCLIDSAvg(FedAvg):
    def __init__(
            self,
            grid: Grid,
            context: Context,
            num_rounds: int,
            fraction_train: float,
            fraction_eval: float,
    ) -> None:
        super().__init__(fraction_train, fraction_eval)
        self.grid = grid
        self.context = context
        self.window_len = cast(int, context.run_config['n-aggregations'])
        self.num_rounds = num_rounds
        self.train_node_ids: list[int] = []
        self.evaluate_node_ids: list[int] = []
        self.all_node_ids: list[int] = []
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        self.timeout = 3600

    # Changed sample_nodes function only to initialize
    def configure_train(
            self,
            server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
    ) -> Iterable[Message]:
        """Configure the next round of federated training"""
        if self.fraction_train == 0.0:
            return []
        if not self.train_node_ids:
            num_nodes = int(len(list(self.grid.get_node_ids())) * self.fraction_train)
            sample_size = max(num_nodes, self.min_train_nodes)
            self.train_node_ids, self.all_node_ids = sample_nodes(
                grid=self.grid,
                min_available_nodes=self.min_available_nodes,
                sample_size=sample_size
            )
        log(
            INFO, "configure_train: Sampled %s nodes (out of %s)",
            len(self.train_node_ids), len(self.all_node_ids)
        )
        config['server-round'] = server_round

        record = RecordDict({
            self.arrayrecord_key: arrays,
            self.configrecord_key: config
        })
        messages = self._construct_messages(
            record=record,
            node_ids=self.train_node_ids,
            message_type=MessageType.TRAIN
        )
        messages = cast(list[Message], messages)
        messages[0].content['config']['profile_on'] = 1
        return messages

    def configure_evaluate(
            self, server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation"""
        if self.fraction_evaluate == 0.0:
            return []

        if not self.evaluate_node_ids:
            num_nodes = int(len(list(self.grid.get_node_ids())) * self.fraction_evaluate)
            sample_size = max(num_nodes, self.min_evaluate_nodes)
            node_ids, _ = sample_nodes(self.grid, self.min_available_nodes, sample_size)
            self.evaluate_node_ids = node_ids
        log(
            INFO, "configure_evaluate: Sampled %s nodes (out of %s)",
            len(self.evaluate_node_ids), len(self.all_node_ids)
        )

        # Always inject current server round
        config['server-round'] = server_round

        # Construct messages
        record = RecordDict({
            self.arrayrecord_key: arrays,
            self.configrecord_key: config
        })
        messages = self._construct_messages(
            record=record,
            node_ids=self.evaluate_node_ids,
            message_type=MessageType.EVALUATE
        )
        return messages

    def _construct_messages(
            self,
            record: RecordDict,
            node_ids: list[int],
            message_type: str
    ) -> list[Message]:
        """Construct N Messages carrying the same RecordDict payload."""
        messages = []
        flows = cast(
            list[list[int]],
            json.loads(cast(bytes, record[self.configrecord_key]['flows']))
        )
        for node_id, flow_list in zip(node_ids, flows):
            record_copy = record.copy()
            config_copy = cast(ConfigRecord, record[self.configrecord_key].copy())
            config_copy['flows'] = flow_list
            record_copy[self.configrecord_key] = config_copy
            messages.append(Message(record_copy, node_id, message_type))
        return messages

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

        aggregated_model: OrderedDict[str, Tensor] = OrderedDict()
        # This iterates over all keys and averages a param's tensor
        with torch.no_grad():
            # Parameter key examples: 'network.9.weight', 'network.9.bias', ...
            # Tensors already on cpu #sus 🤨
            for key in arrays[0].keys():
                tensors_of_key = torch.stack([array[key] for array in arrays])
                averaged_tensor = torch.mean(tensors_of_key, dim=0).cpu()
                aggregated_model[key] = averaged_tensor

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(
            reply_contents,
            self.weighted_by_key,
        )
        return ArrayRecord(aggregated_model), metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            replies: Iterable[Message]
    ) -> Optional[MetricRecord]:
        """Aggregate MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            # Aggregate MetricRecords
            metrics = aggregate_metricrecords(reply_contents, self.weighted_by_key)
        return metrics

    def start(
        self,
        initial_arrays: ArrayRecord,
        current_day: int,
        previous_roc: Optional[float],
        train_config: Optional[list[ConfigRecord]] = None,
        evaluate_config: Optional[list[ConfigRecord]] = None
    ) -> Result:
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds=self.num_rounds,
            arrays=initial_arrays,
            train_config=train_config,
            evaluate_config=evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = train_config or []
        evaluate_config = evaluate_config or []
        result = Result()

        # Variables for recovery time to 95%
        left_pointer = 1 # Points at day 1
        total_auroc = 0.0
        result.evaluate_metrics_clientapp[-1] = MetricRecord({
            'recovery-seconds': -1.0,
            'recovery-round': -1
        })

        t_start = time.time()
        current_array = initial_arrays
        for current_round in range(1, self.num_rounds + 1):
            log(INFO, "")
            log(
                INFO, "[DAY %s | ROUND %s/%s]",
                current_day, current_round, self.num_rounds
            )

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = self.grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    current_array,
                    train_config,
                    self.grid,
                ),
                timeout=self.timeout
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
            evaluate_replies: Iterable[Message] = self.grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    current_array,
                    evaluate_config,
                    self.grid,
                ),
                timeout=self.timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )
            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(
                    INFO, "\t└──> Aggregated MetricRecord: %s",
                    agg_evaluate_metrics
                )
                eval_metrics = result.evaluate_metrics_clientapp
                eval_metrics[current_round] = agg_evaluate_metrics
                # current_round acts as the right pointer of the window,
                # left pointer will shrink (and remove itself)
                if eval_metrics[-1]['recovery-round'] == -1 and current_day > 1:
                    total_auroc += cast(float, agg_evaluate_metrics['auroc'])
                    if current_round > self.window_len:
                        left_metric = eval_metrics[left_pointer]
                        total_auroc -= cast(float, left_metric['auroc'])
                        left_pointer += 1
                    current_auroc = total_auroc / self.window_len
                    auroc_threshold = 0.95 * cast(float, previous_roc)
                    if (current_round >= self.window_len
                        and current_auroc >= auroc_threshold):
                        recovery_time = time.time() - t_start
                        eval_metrics[-1]['recovery-seconds'] = recovery_time
                        eval_metrics[-1]['recovery-round'] = current_round

        # -----------------------------------------------------------------
        # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
        # -----------------------------------------------------------------

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")
        return result


def aggregate_metricrecords(
    records: list[RecordDict],
    weighting_metric_name: str
) -> MetricRecord:
    """
    Modified weighted aggregation of MetricRecords using a specific key.
    Separates the main aggregation and the partial aggregation of epsilon
    for Fed-CL-IDS.
    """
    metric_records = [
        # Get the first (and only) MetricRecord in the record
        next(iter(record.metric_records.values()))
        for record in records
    ]
    metric_records = cast(list[dict[str, float]], metric_records)
    weights = [
        cast(int, metric_dict[weighting_metric_name])
        for metric_dict in metric_records
    ]
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]
    aggregated_metrics = _aggregation(
        metric_records=metric_records,
        weight_factors=weight_factors,
        ignored_keys=(weighting_metric_name, 'epsilon')
    )

    # Writes the epsilon value since the other function would aggregate
    # the sentinel value (-1)
    epsilon_metrics = [
        metric_dict
        for metric_dict in metric_records
        if metric_dict['epsilon'] >= 0
    ]
    epsilon_weights = [
        metric_dict[weighting_metric_name]
        for metric_dict in epsilon_metrics
    ]
    total_epsilon_weight = sum(epsilon_weights)
    epsilon_weight_factors = [w / total_epsilon_weight for w in epsilon_weights]
    # Empty if differential privacy is disabled
    aggregated_epsilon = _aggregation(
        metric_records=epsilon_metrics,
        weight_factors=epsilon_weight_factors,
        ignored_keys=(weighting_metric_name,)
    )

    aggregated_metrics['epsilon'] = aggregated_epsilon.get('epsilon', -1)
    return aggregated_metrics

def _aggregation(
        metric_records: Iterable[MetricRecord],
        weight_factors: Iterable[float],
        ignored_keys: Container[str]
) -> MetricRecord:
    """Aggregates MetricRecords in respect to given weights"""
    aggregated_metrics = MetricRecord()
    for metric_dict, weight in zip(metric_records, weight_factors):
        for key, value in metric_dict.items():
            if key in ignored_keys:
                continue
            default_type = list[float] if isinstance(value, list) else float
            aggregated_value = aggregated_metrics.get(key, default_type())
            if isinstance(value, list):
                aggregated_value = cast(list[float], aggregated_value)
                aggregated_value = aggregated_value or [0.0] * len(value)
                aggregated_metrics[key] = [
                    agg_val + val * weight
                    for agg_val, val in zip(aggregated_value, value)
                ]
                continue
            agg_val = cast(float, aggregated_value)
            aggregated_metrics[key] = agg_val + value * weight
    return aggregated_metrics

def log_strategy_start_info(
    num_rounds: int,
    arrays: ArrayRecord,
    train_config: Optional[list[ConfigRecord]],
    evaluate_config: Optional[list[ConfigRecord]],
) -> None:
    """
    Log information about the strategy start. Modified to prevent
    printing list of flows.
    """
    log(INFO, "\t├── Number of rounds: %d", num_rounds)
    log(
        INFO, "\t├── ArrayRecord (%.2f MB)",
        sum(len(array.data) for array in arrays.values()) / (1024**2),
    )
    log(
        INFO, "\t├── ConfigRecord (train): %s",
        str_config(train_config[0]) if train_config else "(empty!)",
    )
    log(
        INFO, "\t├── ConfigRecord (evaluate): %s",
        str_config(evaluate_config[0]) if evaluate_config else "(empty!)",
    )

def str_config(config: ConfigRecord) -> str:
    """Ensures start log does not print all elements in flows"""
    all_config_str: list[str] = []
    for key, value in config.items():
        if key == 'flows':
            value = json.loads(cast(bytes, value))
        if isinstance(value, bytes):
            string = f"'{key}': '<bytes>'"
        elif isinstance(value, list):
            string = f"'{key}': length = {len(value)}"
        else:
            string = f"'{key}': '{value}'"
        all_config_str.append(string)
    content = ', '.join(all_config_str)
    return f"{{{content}}}"
