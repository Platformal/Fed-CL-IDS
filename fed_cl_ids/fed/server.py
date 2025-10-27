from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from fed_cl_ids.fed.CustomStrategies import UAVIDSFedAvg
from fed_cl_ids.models.mlp import MLP
import torch
import hashlib
import json
import yaml

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Get configurations
    fraction_train = float(context.run_config['fraction-train'])
    fraction_eval = float(context.run_config['fraction-evaluate'])
    n_rounds = int(context.run_config['n-rounds'])
    max_days = int(context.run_config['max-days'])
    n_features = int(context.run_config['n-features'])
    n_classes = int(context.run_config['n-classes'])
    model_width = str(context.run_config['mlp-width'])
    model_dropout = float(context.run_config['mlp-dropout'])
    model_weight_decay = float(context.run_config['mlp-weight-decay'])
    lr_max = float(context.run_config['lr-max'])
    lr_min = float(context.run_config['lr-min'])

    # Create and initialize central model to none.
    central_model = MLP(
        n_features=n_features,
        n_classes=n_classes,
        hidden_widths=[int(x) for x in model_width.split(',')],
        dropout=model_dropout,
        weight_decay=model_weight_decay,
        lr_max=lr_max,
        lr_min=lr_min
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    central_model.to(device)

    n_train_clients = int(len(list(grid.get_node_ids())) * fraction_train)

    # Range of days with list of flows for the day
    uavids_days = yaml.safe_load(open("fed_cl_ids/data_pipeline/splits/uavids_days.yaml"))
    uavids_days = dict(list(uavids_days.items())[:max_days])
    
    # Sampled clients are static and won't change throughout days.
    # Strategy needs to be in loop. Reassigns sampled clients if need be.
    strategy = UAVIDSFedAvg(fraction_train, fraction_eval)

    initial_model = ArrayRecord(central_model.state_dict())
    current_model: ArrayRecord = initial_model
    for day in range(1, max_days + 1):
        # Assign each flow to available clients for given day
        client_map = [[] for _ in range(n_train_clients)]
        for flow_id in uavids_days[f"Day{day}"]:
            id_encoding = str(flow_id).encode()
            id_hash = hashlib.sha256(id_encoding).hexdigest()
            i = int(id_hash, 16) % n_train_clients
            client_map[i].append(flow_id)
        
        flows = ConfigRecord({'flows': json.dumps(client_map)})
        result = strategy.start(
            grid=grid,
            initial_arrays=current_model,
            current_day=day,
            num_rounds=n_rounds,
            train_config=flows,
            evaluate_config=flows,
        )
        current_model = result.arrays
        model_dict = result.arrays.to_torch_state_dict()
        torch.save(model_dict, f"fed_cl_ids/outputs/Day{day}.pt")
        metrics = result.evaluate_metrics_clientapp.popitem()
        with open("fed_cl_ids/outputs/metrics.txt", 'a') as file:
            file.write(f"Day {day}:\n{str(metrics)}\n\n")
