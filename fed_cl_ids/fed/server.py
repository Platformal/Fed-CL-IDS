"""Fed-CL-IDS: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from fed_cl_ids.fed.CustomStrategies import UAVIDSFedAvg
from fed_cl_ids.models.mlp import MLP
from fed_cl_ids.data_pipeline.uavids_preprocess import generate_clients
import json
import pandas
import yaml

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Get configurations
    fraction_train = context.run_config['fraction-train']
    fraction_eval = context.run_config['fraction-evaluate']
    n_rounds = context.run_config['num-server-rounds']
    max_days = context.run_config['max-days']
    model_width = context.run_config['mlp-width']
    model_dropout = context.run_config['mlp-dropout']
    model_weight_decay = context.run_config['mlp-weight-decay']
    lr_max = context.run_config['lr-max']
    lr_min = context.run_config['lr-min']


    learning_rate = 0.02

    # Create and initialize central model to none.
    central_model = MLP(
        n_features=18,
        n_classes=2,
        hidden_widths=[int(x) for x in model_width.split(',')],
        dropout=model_dropout,
        weight_decay=model_weight_decay,
        lr_max=lr_max,
        lr_min=lr_min
    )
    model_params = ArrayRecord(central_model.state_dict())

    # Rehash to N clients > clients.yaml
    n_train_clients = int(len(list(grid.get_node_ids())) * fraction_train)
    # generate_clients(n_train_clients)
    # clients = yaml.safe_load(open("fed_cl_ids/data_pipeline/splits/clients.yaml"))
    # for client_id, flows in clients.items():
    #     clients[client_id] = set(flows)

    # Range of days with list of flows for the day
    days = yaml.safe_load(open("fed_cl_ids/data_pipeline/splits/uavids_days.yaml"))
    days = dict(list(days.items())[:max_days])
    
    # Hash FlowIDs to client
    client_map = [[] for _ in range(n_train_clients)]
    for flow_id in days['Day1']:
        i = hash(flow_id) % n_train_clients
        client_map[i].append(flow_id)  

    # Initialize FedAvg strategy
    strategy = UAVIDSFedAvg(fraction_train, fraction_eval)
    flows = ConfigRecord({'flows': json.dumps(client_map)})
    result = strategy.start(
        grid,
        model_params,
        n_rounds,
        train_config=flows,
        evaluate_config=flows
    )

    # Save final model to disk
    print("\nSaving final model as final_model.pt")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
