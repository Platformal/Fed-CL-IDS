"""Starts federated learning from simulation and configuration file"""
from typing import Optional, Iterable, cast
from collections import OrderedDict
from pathlib import Path
import hashlib
import time

import pandas as pd
from torch import Tensor
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from sklearn.model_selection import train_test_split
from fed.fed_cl_ids_strategies import FedCLIDSModel
from models.fed_metrics import FedMetrics
from models.mlp import MLP

DATA_PIPELINE = Path('data_pipeline')
UAVIDS_DATA_PATH = DATA_PIPELINE / 'preprocessed_uavids'
CICIDS_DATA_PATH = DATA_PIPELINE / 'preprocessed_cicids'
RUNTIME_PATH = Path('runtime')
OUTPUT_PATH = Path('outputs')
METRICS_PATH = OUTPUT_PATH / 'metrics.txt'

class ServerConfiguration:
    """Stores configurations from pyproject.toml"""
    def __init__(self, grid: Grid, context: Context) -> None:
        self.fraction_train = cast(float, context.run_config['fraction-train'])
        self.fraction_evaluate = cast(float, context.run_config['fraction-evaluate'])
        self.total_clients = len(list(grid.get_node_ids()))
        self.n_train_clients = int(self.total_clients * self.fraction_train)
        self.n_evaluate_clients = int(self.total_clients * self.fraction_evaluate)

        self.n_days = cast(int, context.run_config['days'])
        self.n_aggregate = cast(int, context.run_config['n-aggregations'])
        self.n_rounds = cast(int, context.run_config['rounds'])
        self.dp_enabled = cast(bool, context.run_config['dp-enabled'])

class Server:
    """Main class holding configurations, main model parameters, 
    and federated aggregation method."""
    def __init__(self, grid: Grid, context: Context) -> None:
        self.config = ServerConfiguration(grid, context)
        self.federated_model = FedCLIDSModel(
            grid=grid,
            context=context,
            num_rounds=self.config.n_rounds,
            fraction_train=self.config.fraction_train,
            fraction_eval=self.config.fraction_evaluate
        )
        self.current_parameters = self._initial_parameters(context)
        self.total_epsilon = 0.0
        self.dataframe: pd.DataFrame
        self.dataframe_path: Optional[Path] = None

    def _initial_parameters(self, context: Context) -> OrderedDict[str, Tensor]:
        widths = cast(str, context.run_config['mlp-widths'])
        model = MLP(
            n_features=cast(int, context.run_config['n-features']),
            hidden_widths=map(int, widths.split(',')),
            dropout=cast(float, context.run_config['mlp-dropout']),
            weight_decay=cast(float, context.run_config['mlp-weight-decay']),
            lr_max=cast(float, context.run_config['mlp-lr-max']),
            lr_min=cast(float, context.run_config['mlp-lr-min'])
        )
        return cast(OrderedDict, model.state_dict())

    def aggregate_records(self, metric_records: list[MetricRecord]) -> dict[str, float]:
        "Standard arithmetic averaging of given MetricRecords"
        n_recent = len(metric_records)
        metrics = cast(list[dict[str, float]], metric_records)
        eval_metrics = {
            key: round(sum(record[key] for record in metrics) / n_recent, 6)
            for key in metrics[0].keys()
        }
        return eval_metrics

    def log_results(
            self,
            day: int,
            all_rounds: list[MetricRecord],
            agg_metrics: dict[str, float],
            recovery_metric: dict[str, float]
    ) -> None:
        """Saves aggregated model as pt file and logs aggregated metrics"""
        # Saves most recent aggregated data from the rounds not average per day
        torch.save(self.current_parameters, OUTPUT_PATH / f'day{day}.pt')

        day_epsilon = 0.0
        if self.config.dp_enabled:
            rounds = cast(list[dict[str, float]], all_rounds)
            day_epsilon = sum(map(lambda record: record['epsilon'], rounds))
            self.total_epsilon += day_epsilon

        values = {
            # key: (value, rounding, sentinel_value)
            'daily_epsilon': (day_epsilon, 6, self.config.dp_enabled),
            'total_epsilon': (self.total_epsilon, 6, self.config.dp_enabled),
            'recovery-seconds': (recovery_metric['recovery-seconds'], 3, -1),
            'recovery-round': (recovery_metric['recovery-round'], 0, -1)
        }
        formatted_values = self._format_values(values)
        text = [
            f"Day {day}",
            f"{self.config.n_train_clients}/{self.config.total_clients} train",
            f"Rounds: {self.config.n_rounds}",
            f"Day Epsilon: {formatted_values['daily_epsilon']}, "
            f"Total Epsilon: {formatted_values['total_epsilon']}",
            f"Recovery Time (sec): {formatted_values['recovery-seconds']}, "
            f"Recovery Rounds: {formatted_values['recovery-round']}",
            f"{agg_metrics}\n"
        ]
        with METRICS_PATH.open('a', encoding='utf-8') as file:
            file.write(' | '.join(text))
            if day == self.config.n_days:
                file.write('\n')

    def _format_values(
            self,
            values: dict[str, tuple[float, int, bool | int | float]]
    ) -> dict[str, Optional[float]]:
        new_dict: dict[str, Optional[float]] = {}
        for key, (value, rounding, sentinel_value) in values.items():
            if isinstance(sentinel_value, bool) and not sentinel_value:
                value = None
            elif isinstance(sentinel_value, (int, float)) and value == sentinel_value:
                value = None
            else:
                value = round(value, rounding)
            new_dict[key] = value
        return new_dict

    def get_data(
            self,
            filepath: Path
    ) -> tuple[list[ConfigRecord], list[ConfigRecord]]:
        """Splits flows into train and evaluate and distributes them among
        n_clients for each respective flow"""
        if not filepath.exists():
            raise FileNotFoundError(f"Cannot find file name: {filepath}")
        # 80/20 split
        train_split, eval_split = self._split_data(pd.read_parquet(filepath))
        # Train flows (80%) gets hashed into 5 clients (train_pf = 0.5)
        # Evaluate flows (20%) gets hashed into 10 clients (evaluate_pf = 1.0)
        train_tasks, eval_tasks = map(
            self._distribute_flows,
            (train_split, eval_split),
            (self.config.n_train_clients, self.config.n_evaluate_clients)
        )
        train_records = [
            ConfigRecord({'flows': train_flows, 'filepath': str(filepath)})
            for train_flows in train_tasks
        ]
        eval_records = [
            ConfigRecord({'flows': eval_flows, 'filepath': str(filepath)})
            for eval_flows in eval_tasks
        ]
        return train_records, eval_records

    def _split_data(
            self,
            dataframe: pd.DataFrame,
            train_ratio: float = 0.8,
            random_seed: Optional[int] = None
    ) -> tuple[list[int], list[int]]:
        """Splits a list of flow IDs into two datasets."""
        labels = dataframe['label']
        splits = train_test_split(
            list(range(len(dataframe))),
            train_size=train_ratio,
            random_state=random_seed,
            stratify=labels
        )
        return tuple(splits)

    def _distribute_flows(self, flows: Iterable[int], n_clients: int) -> list[list[int]]:
        """Hash each flow by ID and assign to a bucket for each client."""
        clients = [[] for _ in range(n_clients)]
        for flow_id in flows:
            id_bytes = str(flow_id).encode()
            id_hash = hashlib.sha256(id_bytes).hexdigest()
            i = int(id_hash, 16) % n_clients
            clients[i].append(flow_id)
        return clients

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Triggered when flwr run is called."""
    server = Server(grid, context)
    if RUNTIME_PATH.exists():
        clear_directory(RUNTIME_PATH)
    else:
        RUNTIME_PATH.mkdir()

    daily_metric_logs: list[list[MetricRecord]] = []
    previous_roc: Optional[float] = None
    mapped_filepath = {
        day: CICIDS_DATA_PATH / f"{day}.parquet"
        for day in range(1, server.config.n_days + 1)
    }

    start = time.time()
    for day, filepath in mapped_filepath.items():
        train_flows, evaluate_flows = server.get_data(filepath)
        daily_result = server.federated_model.start(
            initial_arrays=ArrayRecord(server.current_parameters),
            current_day=day,
            previous_roc=previous_roc,
            train_config=train_flows,
            evaluate_config=evaluate_flows
        )
        server.current_parameters = daily_result.arrays.to_torch_state_dict()

        all_rounds = daily_result.evaluate_metrics_clientapp
        recovery_metric = cast(dict[str, float], all_rounds.pop(-1))
        clean_metrics = list(all_rounds.values())

        daily_metric_logs.append(clean_metrics)
        recent_metrics = clean_metrics[-server.config.n_aggregate:]
        agg_metrics = server.aggregate_records(recent_metrics)
        previous_roc = agg_metrics['auroc']
        server.log_results(day, clean_metrics, agg_metrics, recovery_metric)

    print(f"Final Time: {time.time() - start:.3f}")
    FedMetrics.create_metric_plots(daily_metric_logs, OUTPUT_PATH)
    clear_directory(RUNTIME_PATH)

def clear_directory(filepath: Path) -> None:
    """Removes all files in a folder directory"""
    for file in filter(Path.is_file, filepath.iterdir()):
        file.unlink()
