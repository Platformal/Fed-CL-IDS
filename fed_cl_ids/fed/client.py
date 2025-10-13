"""Fed-CL-IDS: A Flower / PyTorch app."""

import torch
import pandas as pd
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from fed_cl_ids.models.mlp import MLP, load_data, split_uavids
from fed_cl_ids.models.mlp import test as test_fn
from fed_cl_ids.models.mlp import train as train_fn
from fed_cl_ids.models.mlp import client_train, client_test

# Flower ClientApp
app = ClientApp()

dataset: pd.DataFrame | None = None

# Needs flows for specific client for specific day
# Client needs ID to receive correct flows
# Client will load and store the dataset globally
@app.train()
def train(msg: Message, context: Context) -> Message:
    model_width = context.run_config['mlp-width']
    model = MLP(
        18,
        2,
        [int(x) for x in model_width.split(',')],
        context.run_config['mlp-dropout'],
        context.run_config['mlp-weight-decay'],
        context.run_config['lr-max'],
        context.run_config['lr-min']
    )
    model_params = msg.content['arrays'].to_torch_state_dict()
    model.load_state_dict(model_params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    flow_ids: list[int] = msg.content['config']['flows']
    uavids_path = "fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv"
    global dataset
    if dataset is None:
        dataset = pd.read_csv(uavids_path).set_index('FlowID')
    train, _ = split_uavids(dataset, flow_ids)
    epochs = context.run_config['local-epochs']
    avg_loss = client_train(model, train, epochs, device)
    client_model_params = ArrayRecord(model.state_dict())
    metrics = {
        'train_loss': avg_loss,
        'num-examples': len(train)
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({
        'arrays': client_model_params, 
        'metrics': metric_record
    })
    return Message(content=content, reply_to=msg)
    

# @app.train()
# def train(msg: Message, context: Context) -> Message:
#     """Train the model on local data."""

#     # Load the model and initialize it with the received weights
#     model = MLP()
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     trainloader, _ = load_data(partition_id, num_partitions)

#     # Call the training function
#     train_loss = train_fn(
#         model,
#         trainloader,
#         context.run_config["local-epochs"],
#         msg.content["config"]["lr"],
#         device,
#     )

#     # Construct and return reply Message
#     model_record = ArrayRecord(model.state_dict())
#     metrics = {
#         "train_loss": train_loss,
#         "num-examples": len(trainloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"arrays": model_record, "metrics": metric_record})
#     return Message(content=content, reply_to=msg)


# ROC-AUC, PR-AUC, macro-F1, Recall@FPR = 1%
@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    model_width = context.run_config['mlp-width']
    model = MLP(
        18,
        2,
        [int(x) for x in model_width.split(',')],
        context.run_config['mlp-dropout'],
        context.run_config['mlp-weight-decay'],
        context.run_config['lr-max'],
        context.run_config['lr-min']
    )
    model_params = msg.content['arrays'].to_torch_state_dict()
    model.load_state_dict(model_params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    flow_ids: list[int] = msg.content['config']['flows']
    uavids_path = "fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv"
    global dataset
    if dataset is None:
        dataset =  pd.read_csv(uavids_path).set_index('FlowID')
    _, test = split_uavids(dataset, flow_ids)

    eval_loss, eval_acc = client_test(model, test, device)
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(test),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)


# @app.evaluate()
# def evaluate(msg: Message, context: Context) -> Message:
#     """Evaluate the model on local data."""

#     # Load the model and initialize it with the received weights
#     model = MLP()
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     _, valloader = load_data(partition_id, num_partitions)

#     # Call the evaluation function
#     eval_loss, eval_acc = test_fn(
#         model,
#         valloader,
#         device,
#     )

#     # Construct and return reply Message
#     metrics = {
#         "eval_loss": eval_loss,
#         "eval_acc": eval_acc,
#         "num-examples": len(valloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"metrics": metric_record})
#     return Message(content=content, reply_to=msg)
