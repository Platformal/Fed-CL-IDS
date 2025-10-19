import torch
import pandas as pd
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from fed_cl_ids.models.mlp import MLP, split_uavids
from fed_cl_ids.models.mlp import client_train, client_test

# Flower ClientApp
app = ClientApp()

dataset: pd.DataFrame | None = None

# Server sends message of array dict, initialize and load model parameters.
def load_server_model(msg: Message, context: Context, device: torch.device) -> MLP:
    widths: str = context.run_config['mlp-width']
    model = MLP(
        n_features=context.run_config['n-features'],
        n_classes=context.run_config['n-classes'],
        hidden_widths=[int(x) for x in widths.split(',')],
        dropout=context.run_config['mlp-dropout'],
        weight_decay=context.run_config['mlp-weight-decay'],
        lr_max=context.run_config['lr-max'],
        lr_min=context.run_config['lr-min']
    )
    model_params = msg.content['arrays'].to_torch_state_dict()
    model.load_state_dict(model_params)
    model.to(device)
    return model

# Client MLP model needs to receive initial parameters to construct model.
@app.train()
def train(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_server_model(msg, context, device)
    
    flow_ids: list[int] = msg.content['config']['flows']
    uavids_path = "fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv"
    global dataset
    if dataset is None:
        dataset = pd.read_csv(uavids_path).set_index('FlowID')
    batch_size: int = context.run_config['batch-size']
    train, _ = split_uavids(dataset, flow_ids, batch_size)
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

# ROC-AUC, PR-AUC, macro-F1, Recall@FPR = 1%
@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_server_model(msg, context, device)

    flow_ids: list[int] = msg.content['config']['flows']
    uavids_path = "fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv"
    global dataset
    if dataset is None:
        dataset =  pd.read_csv(uavids_path).set_index('FlowID')
    batch_size: int = context.run_config['batch-size']
    _, test = split_uavids(dataset, flow_ids, batch_size)

    eval_loss, eval_acc = client_test(model, test, device)
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(test),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
