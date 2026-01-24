from typing import Optional, Iterable, cast
from collections import OrderedDict
from logging import INFO
from io import StringIO
import time
import math

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

class FedCLIDSModel(FedAvg):
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
        self.num_rounds = num_rounds
        self.train_nodes: list[int] = []
        self.evaluate_nodes: list[int] = []
        self.all_nodes: list[int] = list(grid.get_node_ids())
        self.timeout = 3600

    # Changed sample_nodes function only to initialize
    def configure_train(
            self,
            arrays: ArrayRecord,
            configs: Iterable[ConfigRecord]
    ) -> list[Message]:
        """Configure the next round of federated training"""
        if self.fraction_train == 0.0:
            return []
        log(
            INFO, "configure_train: Sampled %s nodes (out of %s)",
            len(self.train_nodes), len(self.all_nodes)
        )
        messages = self._construct_messages(
            arrays=arrays,
            node_ids=self.train_nodes,
            message_type=MessageType.TRAIN,
            configs=configs
        )
        return messages

    def configure_evaluate(
            self,
            arrays: ArrayRecord,
            configs: list[ConfigRecord]
    ) -> list[Message]:
        """Configure the next round of federated evaluation"""
        if self.fraction_evaluate == 0.0:
            return []
        log(
            INFO, "configure_evaluate: Sampled %s nodes (out of %s)",
            len(self.evaluate_nodes), len(self.all_nodes)
        )
        messages = self._construct_messages(
            arrays=arrays,
            node_ids=self.evaluate_nodes,
            message_type=MessageType.EVALUATE,
            configs=configs
        )
        return messages

    def sample_nodes(self) -> None:
        """
        Modifies node IDs of train, evaluate, and all node IDs.
        If resampling enabled, it will randomly select train and evaluate
        nodes again if it isn't the first time sampling.
        """
        if self.train_nodes or self.evaluate_nodes:
            return
        n_all_nodes = len(self.all_nodes)
        if self.fraction_evaluate:
            n_train = int(self.fraction_train * n_all_nodes)
            self.train_nodes, _ = sample_nodes(
                grid=self.grid,
                min_available_nodes=self.min_available_nodes,
                sample_size=max(n_train, self.min_train_nodes)
            )
        if self.fraction_evaluate:
            n_evaluate = int(self.fraction_evaluate * n_all_nodes)
            self.evaluate_nodes, _ = sample_nodes(
                grid=self.grid,
                min_available_nodes=self.min_available_nodes,
                sample_size=max(n_evaluate, self.min_available_nodes)
            )

    def _construct_messages(
            self,
            arrays: ArrayRecord,
            node_ids: list[int],
            message_type: str,
            configs: Iterable[ConfigRecord]
    ) -> list[Message]:
        """Construct N Messages carrying the different RecordDict payloads."""
        messages: list[Message] = []
        for node_id, config_record in zip(node_ids, configs):
            record = RecordDict({
                self.arrayrecord_key: arrays,
                self.configrecord_key: config_record
            })
            messages.append(Message(record, node_id, message_type))
        return messages

    def aggregate_train(
            self,
            replies: Iterable[Message]
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords from response Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None

        reply_contents = [msg.content for msg in valid_replies]
        arrays: list[dict[str, Tensor]] = []
        for content in reply_contents:
            array_record = cast(ArrayRecord, content[self.arrayrecord_key])
            arrays.append(array_record.to_torch_state_dict())

        aggregated_model: OrderedDict[str, Tensor] = OrderedDict()
        # Averages all of a tensor of a key
        with torch.no_grad():
            # Parameter key examples: 'network.9.weight', 'network.9.bias', ...
            for key in arrays[0].keys():
                tensors_of_key = torch.stack([array[key] for array in arrays])
                aggregated_model[key] = torch.mean(tensors_of_key, dim=0).cpu()

        metrics = aggregate_metricrecords(reply_contents, self.weighted_by_key)
        return ArrayRecord(aggregated_model), metrics

    def aggregate_evaluate(self, replies: Iterable[Message]) -> Optional[MetricRecord]:
        """Aggregate MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        if not valid_replies:
            return None
        reply_contents = [msg.content for msg in valid_replies]
        main_metrics = aggregate_metricrecords(
            records=reply_contents,
            weight_key=self.weighted_by_key
        )

        # If NaN in auroc/recall@fpr=1, then filter and reaggregate
        # to get the correct weighted average
        valid_aurocs: list[RecordDict] = []
        for record_dict in reply_contents:
            metric_record = next(iter(record_dict.metric_records.values()))
            if not math.isnan(cast(float, metric_record['auroc'])):
                valid_aurocs.append(record_dict)

        if valid_aurocs:
            non_nan_metrics = aggregate_metricrecords(
                records=valid_aurocs,
                weight_key=self.weighted_by_key
            )
            main_metrics['auroc'] = non_nan_metrics['auroc']
            main_metrics['recall@fpr=1%'] = non_nan_metrics['recall@fpr=1%']
        return main_metrics

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
        log(INFO, '')

        self.sample_nodes()

        train_config = train_config or [
            ConfigRecord()
            for _ in range(len(self.train_nodes))
        ]
        evaluate_config = evaluate_config or [
            ConfigRecord()
            for _ in range(len(self.evaluate_nodes))
        ]
        result = Result()

        # Variables for recovery time to 95%
        left_pointer = 1 # Points to round 1
        total_auroc = 0.0
        window_len = cast(int, self.context.run_config['n-aggregations'])
        result.evaluate_metrics_clientapp[-1] = MetricRecord({
            'recovery-seconds': -1.0,
            'recovery-round': -1
        })

        t_start = time.time()
        current_array = initial_arrays
        for current_round in range(1, self.num_rounds + 1):
            log(INFO, '')
            log(
                INFO, "[DAY %s | ROUND %s/%s]",
                current_day, current_round, self.num_rounds
            )

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_messages = self.configure_train(current_array, train_config)
            train_replies = self.grid.send_and_receive(
                messages=train_messages,
                timeout=self.timeout
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(train_replies)

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                current_array = agg_arrays
            if agg_train_metrics is not None:
                rounded_metrics = {
                    key: round(value, 6)
                    for key, value in agg_train_metrics.items()
                }
                log(INFO, "\t└──> Aggregated MetricRecord: %s", rounded_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_messages = self.configure_evaluate(
                arrays=current_array,
                configs=evaluate_config
            )
            evaluate_replies: Iterable[Message] = self.grid.send_and_receive(
                messages=evaluate_messages,
                timeout=self.timeout,
            )

            # Aggregate metrics after collecting all metrics from eval nodes.
            agg_evaluate_metrics = self.aggregate_evaluate(evaluate_replies)
            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                rounded_metrics = {
                    key: round(value, 6)
                    for key, value in agg_evaluate_metrics.items()
                }
                log(INFO, "\t└──> Aggregated MetricRecord: %s", rounded_metrics)

                eval_metrics = result.evaluate_metrics_clientapp
                epsilon = cast(MetricRecord, agg_train_metrics)['epsilon']
                agg_evaluate_metrics['epsilon'] = epsilon
                eval_metrics[current_round] = agg_evaluate_metrics

                # current_round acts as the right pointer of the window,
                # left pointer will shrink (and remove itself)
                if eval_metrics[-1]['recovery-round'] == -1 and current_day > 1:
                    total_auroc += cast(float, agg_evaluate_metrics['auroc'])
                    if current_round > window_len:
                        left_metric = eval_metrics[left_pointer]
                        total_auroc -= cast(float, left_metric['auroc'])
                        left_pointer += 1
                    current_auroc = total_auroc / window_len
                    auroc_threshold = 0.95 * cast(float, previous_roc)
                    if (current_round >= window_len
                        and current_auroc >= auroc_threshold):
                        recovery_time = time.time() - t_start
                        eval_metrics[-1]['recovery-seconds'] = recovery_time
                        eval_metrics[-1]['recovery-round'] = current_round

        log(INFO, '')
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, '')
        log(INFO, "Final results:")
        log(INFO, '')
        for line in StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, '')
        return result


def aggregate_metricrecords(
    records: list[RecordDict],
    weight_key: str,
) -> MetricRecord:
    """
    Modified weighted aggregation of MetricRecords using a specific key.
    Separates the main aggregation and the partial aggregation of epsilon
    for Fed-CL-IDS.
    """
    metric_records = [
        next(iter(record.metric_records.values()))
        for record in records
    ]
    weights = [
        cast(int, metric_dict[weight_key])
        for metric_dict in metric_records
    ]
    total_weight = sum(weights)
    aggregated_metrics = _aggregation(
        metric_records=metric_records,
        weight_factors=[w / total_weight for w in weights],
        weight_key=weight_key
    )
    return aggregated_metrics

def _aggregation(
        metric_records: Iterable[MetricRecord],
        weight_factors: Iterable[float],
        weight_key: str
) -> MetricRecord:
    """Aggregates MetricRecords in respect to given weights"""
    aggregated_metrics = MetricRecord()
    for metric_dict, weight in zip(metric_records, weight_factors):
        for key, value in metric_dict.items():
            if key == weight_key:
                continue
            is_list = isinstance(value, list)
            default_type = list[float] if is_list else float
            aggregated_value = aggregated_metrics.get(key, default_type())
            if is_list:
                aggregated_value = cast(list[float], aggregated_value)
                aggregated_value = aggregated_value or [0.0] * len(value)
                aggregated_metrics[key] = [
                    current_val + val * weight
                    for current_val, val in zip(aggregated_value, value)
                ]
                continue
            current_val = cast(float, aggregated_value)
            aggregated_metrics[key] = current_val + value * weight
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
    """Prevent printing all elements from flows (list) in ConfigRecord"""
    all_config_str: list[str] = []
    for key, value in config.items():
        if isinstance(value, bytes):
            string = f"'{key}': '<bytes>'"
        elif isinstance(value, list):
            string = f"'{key}': length = {len(value)}"
        else:
            string = f"'{key}': '{value}'"
        all_config_str.append(string)
    content = ', '.join(all_config_str)
    return f"{{{content}}}"
