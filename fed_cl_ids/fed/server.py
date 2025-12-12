"""Starts federated learning from simulation and configuration file"""
from typing import Optional, Iterable
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

from torch import Tensor
import torch
import yaml
import pandas as pd

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp.strategy import Result
from flwr.serverapp import Grid, ServerApp

from sklearn.model_selection import train_test_split
from fed_cl_ids.fed.custom_strategies import UAVIDSFedAvg
from fed_cl_ids.models.mlp import MLP

MAIN_PATH = Path().cwd() / 'fed_cl_ids'
UAVIDS_DAYS_PATH = MAIN_PATH / 'data_pipeline' / 'splits' / 'uavids_days.yaml'
UAVIDS_DATA_PATH = MAIN_PATH / 'datasets' / 'UAVIDS-2025 Preprocessed.csv'
RUNTIME_PATH = MAIN_PATH / 'runtime'
OUTPUT_PATH = MAIN_PATH / 'outputs'

@dataclass()
class ServerConfiguration:
    """Stores configurations from pyproject.toml"""
    def __init__(self, grid: Grid, context: Context) -> None:
        self.fraction_train = float(context.run_config['fraction-train'])
        self.fraction_evaluate = float(context.run_config['fraction-evaluate'])
        self.total_clients = len(tuple(grid.get_node_ids()))
        self.n_train_clients = int(self.total_clients * self.fraction_train)
        self.n_evaluate_clients = int(self.total_clients * self.fraction_evaluate)

        self.n_days = int(context.run_config['max-days'])
        self.n_rounds = int(context.run_config['n-rounds'])

class Server:
    """Main class holding server configurations and strategy."""
    def __init__(self, grid: Grid, context: Context) -> None:
        self.config = ServerConfiguration(grid, context)
        self.federated_model = UAVIDSFedAvg(
            fraction_train=self.config.fraction_train,
            fraction_eval=self.config.fraction_evaluate,
        )
        self.current_parameters = self._initial_parameters(context)

        self.dataframe: pd.DataFrame
        self.dataframe_path: Optional[Path] = None

    def _initial_parameters(self, context: Context) -> OrderedDict[str, Tensor]:
        widths = str(context.run_config['mlp-widths'])
        model = MLP(
            n_features=int(context.run_config['n-features']),
            hidden_widths=map(int, widths.split(',')),
            dropout=float(context.run_config['mlp-dropout']),
            weight_decay=float(context.run_config['mlp-weight-decay']),
            lr_max=float(context.run_config['mlp-lr-max']),
            lr_min=float(context.run_config['mlp-lr-min'])
        )
        return OrderedDict(model.state_dict())

    def split_data(
            self,
            raw_flows: list[int],
            train_ratio: float = 0.8,
            random_seed: Optional[int] = None,
            csv_path: Optional[Path] = None) -> list[list[int]]:
        """
        Splits a list of flow IDs into two datasets.
        Can optionally be stratified by passing in csv filepath.
        
        :param raw_flows: Integers for a given day to split into two datasets.
        :type raw_flows: list[int]
        :param train_ratio: Value between 0 and 1 to split training and testing set by.
        :type train_ratio: float
        :param random_seed: Integer for reproducibility.
        :type random_seed: Optional[int]
        :param csv_path: Location of csv file to allow label stratification. 
        Stratifies by the 'label' column.
        :type csv_path: Optional[Path]
        :return: Two lists (from train_test_split) of integers.
        :rtype: list[list[int]]
        """
        labels: Optional[pd.Series] = None
        if csv_path:
            if self.dataframe_path is not None or self.dataframe_path != csv_path:
                self.dataframe_path = csv_path
                self.dataframe = pd.read_csv(
                    filepath_or_buffer=csv_path,
                    dtype={'label': 'uint8'},
                    index_col='FlowID'
                )
            labels = self.dataframe.loc[raw_flows]['label']
        splits = train_test_split(
            raw_flows,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=labels
        )
        return splits

    @staticmethod
    def distribute_flows(flows: Iterable[int], n_clients: int) -> tuple[list[int], ...]:
        """
        Hash each flow by ID and assign to a bucket for each client.
        
        :param flows: Integers representing a flow ID.
        :type flows: list[int]
        :param n_clients: Number of clients to create buckets.
        :type n_clients: int
        :return: List of integers for each client to process.
        :rtype: tuple[list[int], ...]
        """
        clients = tuple([] for _ in range(n_clients))
        for flow_id in flows:
            id_bytes = str(flow_id).encode()
            id_hex = hashlib.sha256(id_bytes).hexdigest()
            i = int(id_hex, 16) % n_clients
            clients[i].append(flow_id)
        return clients

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Triggered when flwr run is called.
    
    :param grid: Description
    :type grid: Grid
    :param context: Description
    :type context: Context
    """

    try:
        RUNTIME_PATH.mkdir()
    except FileExistsError:
        clear_directory(RUNTIME_PATH)

    server = Server(grid, context)
    uavids_days = get_uavids(server, UAVIDS_DAYS_PATH)

    start = time.time()
    for day, raw_flows in enumerate(uavids_days.values(), 1):
        train_flows, evaluate_flows = server.split_data(
            raw_flows=raw_flows,
            csv_path=UAVIDS_DATA_PATH
        )
        train_flows = Server.distribute_flows(
            flows=train_flows,
            n_clients=server.config.n_train_clients
        )
        evaluate_flows = Server.distribute_flows(
            flows=evaluate_flows,
            n_clients=server.config.n_evaluate_clients
        )

        # UAVIDSFedAvg will split flows to each client
        train_config = ConfigRecord({'flows': json.dumps(train_flows)})
        evaluate_config = ConfigRecord({'flows': json.dumps(evaluate_flows)})

        result = server.federated_model.start(
            grid=grid,
            initial_arrays=ArrayRecord(server.current_parameters),
            current_day=day,
            num_rounds=server.config.n_rounds,
            train_config=train_config,
            evaluate_config=evaluate_config,
        )
        log_results(server, result, day)
    with open(OUTPUT_PATH / 'metrics.txt', 'a', encoding='utf-8') as file:
        file.write('\n')
    clear_directory(RUNTIME_PATH)
    print(time.time() - start)

def clear_directory(filepath: Path) -> None:
    """
    Removes all files in a folder directory

    :param path: Path to the folder
    :type path: Path
    """
    for file in [path for path in filepath.iterdir() if path.is_file()]:
        file.unlink()

def get_uavids(server: Server, filepath: Path) -> dict[str, list[int]]:
    """
    Opens form yaml file, and filters days from configuration file
    
    :param server:
    :type server: Server
    :param filepath: Path to yaml file containing flows IDs
    :type filepath: Path
    :return: Contains day as a str and all the flow IDs
    :rtype: dict[str, list[int]]
    """
    with open(filepath, encoding='utf-8') as file:
        raw_days: dict[str, list[int]] = yaml.safe_load(file)
    # Assuming dict is sorted/ordered by days
    filtered_days = tuple(raw_days.items())[:server.config.n_days]
    return dict(filtered_days)

def log_results(server: Server, result: Result, day: int) -> None:
    """Saves aggregated model as pt file and logs aggregated metrics"""
    # I assume the resulting arrays are already in cpu. Remove?
    server.current_parameters = OrderedDict({
        key: state.cpu()
        for key, state in result.arrays.to_torch_state_dict().items()
    })
    torch.save(server.current_parameters, OUTPUT_PATH / f'Day{day}.pt')
    metrics = result.evaluate_metrics_clientapp.popitem()
    with open(OUTPUT_PATH / 'metrics.txt', 'a', encoding='utf-8') as file:
        file.write(
            f"Day {day}: "
            f"{server.config.n_train_clients}/{server.config.total_clients} "
            f"clients: {str(metrics)}\n"
        )
