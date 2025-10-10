"""Fed-CL-IDS: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fed_cl_ids.models.mlp import MLP

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Get configurations
    fraction_train = context.run_config["fraction-train"]
    n_rounds = context.run_config["num-server-rounds"]
    learning_rate = context.run_config["lr"]
    model_width = context.run_config['mlp-width']
    model_dropout = context.run_config['mlp-dropout']
    model_weight_decay = context.run_config['mlp-weight-decay']

    # Create and initialize central model to none.
    central_model = MLP(
        n_features=18,
        n_classes=2,
        hidden_widths=[int(x) for x in model_width.split(',')],
        dropout=model_dropout,
        weight_decay=model_weight_decay,
        lr_max=1e-3,
        lr_min=1e-4
    )
    arrays = ArrayRecord(central_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Start strategy, run FedAvg for num_rounds
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": learning_rate}),
        num_rounds=n_rounds,
    )

    # Save final model to disk
    print("\nSaving final model as final_model.pt")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
