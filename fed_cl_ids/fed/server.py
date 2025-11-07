from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from sklearn.model_selection import train_test_split
from fed_cl_ids.fed.CustomStrategies import UAVIDSFedAvg
from fed_cl_ids.models.mlp import MLP

from typing import Optional
from torch import Tensor
import torch
import pandas as pd
import hashlib
import json
import yaml
import time
import os

class Server:
    def __init__(self, grid: Grid, context: Context) -> None:
        self.fraction_train = float(context.run_config['fraction-train'])
        self.fraction_evaluate = float(context.run_config['fraction-evaluate'])
        self.total_clients = len(list(grid.get_node_ids()))
        self.n_train_clients = int(self.total_clients * self.fraction_train)
        self.n_evaluate_clients = int(self.total_clients * self.fraction_evaluate)

        self.n_days = int(context.run_config['max-days'])
        self.n_rounds = int(context.run_config['n-rounds'])
        self.federated_model = UAVIDSFedAvg(
            fraction_train=self.fraction_train,
            fraction_eval=self.fraction_evaluate
        )
        self.current_parameters = self._model_initial_parameters(context)

        self.dataframe: pd.DataFrame
        self.dataframe_path: Optional[str] = ''

    def _model_initial_parameters(self, context: Context) -> dict[str, Tensor]:
        widths = str(context.run_config['mlp-widths'])
        model = MLP(
            n_features=int(context.run_config['n-features']),
            hidden_widths=[int(x) for x in widths.split(',')],
            dropout=float(context.run_config['mlp-dropout']),
            weight_decay=float(context.run_config['mlp-weight-decay']),
            lr_max=float(context.run_config['mlp-lr-max']),
            lr_min=float(context.run_config['mlp-lr-min'])
        )
        return model.state_dict()

    def split_data(
            self, raw_flows: list[int], 
            train_ratio: float = 0.8,
            random_seed: Optional[int] = None,
            csv_path: Optional[str] = None) -> list[list[int]]:
        '''If passed in a path, proportionally split the multiclass labels
        (stratification) to train and test sets'''
        labels: Optional[pd.Series] = None
        if csv_path:
            if not hasattr(self, 'dataframe') or self.dataframe_path != csv_path:
                self.dataframe = pd.read_csv(csv_path, dtype={'label': 'uint8'})
                self.dataframe = self.dataframe.set_index('FlowID')
                self.dataframe_path = csv_path
            labels = self.dataframe.loc[raw_flows]['label']
        return train_test_split(
            raw_flows, 
            train_size=train_ratio, 
            random_state=random_seed,
            stratify=labels
        )

    @staticmethod
    def distribute_flows(flows: list[int], n_clients: int) -> list[list[int]]:
        clients = [[] for _ in range(n_clients)]
        for flow_id in flows:
            id_bytes = str(flow_id).encode()
            id_hex = hashlib.sha256(id_bytes).hexdigest()
            i = int(id_hex, 16) % n_clients
            clients[i].append(flow_id)
        return clients

def clear_directory(path: str) -> None:
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        os.remove(file_path)

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    server = Server(grid, context)
    
    uavids_path = "fed_cl_ids/data_pipeline/splits/uavids_days.yaml"
    raw_uavids_days = yaml.safe_load(open(uavids_path))
    raw_uavids_days = list(raw_uavids_days.items())[:server.n_days]
    uavids_days: dict[str, list[int]] = dict(raw_uavids_days)

    runtime_path = os.path.join("fed_cl_ids", "runtime")
    clear_directory(runtime_path)

    start = time.time()
    for day, raw_flows in enumerate(uavids_days.values(), 1):
        data_path = "fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv"
        splits = server.split_data(raw_flows, csv_path=data_path)
        train_flows, evaluate_flows = splits
        train_flows = Server.distribute_flows(train_flows, server.n_train_clients)
        evaluate_flows = Server.distribute_flows(evaluate_flows, server.n_evaluate_clients)
        
        # UAVIDSFedAvg will split flows to each client
        train_config = ConfigRecord({'flows': json.dumps(train_flows)})
        evaluate_config = ConfigRecord({'flows': json.dumps(evaluate_flows)})

        result = server.federated_model.start(
            grid=grid,
            initial_arrays=ArrayRecord(server.current_parameters),
            current_day=day,
            num_rounds=server.n_rounds,
            train_config=train_config,
            evaluate_config=evaluate_config
        )

        server.current_parameters = result.arrays.to_torch_state_dict()
        torch.save(server.current_parameters, f"fed_cl_ids/outputs/Day{day}.pt")
        metrics = result.evaluate_metrics_clientapp.popitem()
        with open("fed_cl_ids/outputs/metrics.txt", 'a') as file:
            file.write(
                f"Day {day}: {server.n_train_clients}/{server.total_clients} "
                f"clients: {str(metrics)}\n"
            )
    
    with open("fed_cl_ids/outputs/metrics.txt", 'a') as file:
        file.write('\n')
    clear_directory(runtime_path)
    print(time.time() - start)