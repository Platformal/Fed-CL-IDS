from fed_cl_ids.models.losses import roc_auc, pr_auc, macro_f1, recall_at_fpr
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from sklearn.model_selection import train_test_split
from flwr.clientapp import ClientApp
from fed_cl_ids.models.mlp import MLP
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss
from torch import Tensor
import torch
import random
import pandas as pd
import numpy as np

# Flower ClientApp
app = ClientApp()

dataset: pd.DataFrame | None = None
# Feature tensors, and label tensor
replay_buffer: list[tuple[Tensor, Tensor]] = []
total_flows = 0

# Server sends message of array dict, initialize and load model parameters.
def load_server_model(msg: Message, context: Context, device: torch.device) -> MLP:
    widths = str(context.run_config['mlp-width'])
    model = MLP(
        n_features=int(context.run_config['n-features']),
        n_classes=int(context.run_config['n-classes']),
        hidden_widths=[int(x) for x in widths.split(',')],
        dropout=float(context.run_config['mlp-dropout']),
        weight_decay=float(context.run_config['mlp-weight-decay']),
        lr_max=float(context.run_config['lr-max']),
        lr_min=float(context.run_config['lr-min'])
    )
    model_params = msg.content['arrays'].to_torch_state_dict()
    model.load_state_dict(model_params)
    return model.to(device)

def load_pd_csv(file_path) -> pd.DataFrame:
    global dataset
    if dataset is None:
        dataset = pd.read_csv(file_path).set_index('FlowID')
    return dataset

def update_replay_buffer(train_set: tuple[Tensor, Tensor], replay_ratio: float) -> None:
    new_samples = []
    features, labels = train_set
    for feature, label in zip(features, labels):
        new_samples.append((feature.cpu(), label.cpu()))
    global replay_buffer
    replay_buffer.extend(new_samples)

    max_buffer_size = int(replay_ratio * total_flows)
    if len(replay_buffer) > max_buffer_size:
        replay_buffer = replay_buffer[-max_buffer_size:]

# Client MLP model needs to receive initial parameters to construct model.
@app.train()
def train(msg: Message, context: Context) -> Message:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_server_model(msg, context, device)
    flow_ids: list[int] = msg.content['config']['flows']

    uavids_path = "fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv"
    data = load_pd_csv(uavids_path)
    train_set, _ = split_uavids(data, flow_ids)

    avg_loss = client_train(context, model, train_set, device)
    client_model_params = ArrayRecord(model.state_dict())
    metrics = {
        'train_loss': avg_loss,
        'num-examples': len(train_set)
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({
        'arrays': client_model_params, 
        'metrics': metric_record
    })
    return Message(content=content, reply_to=msg)

def client_train(
        context: Context, model: MLP, train_set: tuple[Tensor, Tensor], 
        device: torch.device) -> float:
    train_features, train_labels = train_set
    total_samples = len(train_labels)

    replay_ratio = float(context.run_config['replay-ratio'])
    replay_size = int(replay_ratio * len(train_features))
    if replay_buffer and replay_size:
        samples = random.sample(
            population=replay_buffer, 
            k=min(replay_size, len(replay_buffer))
        )
        old_features = torch.stack([feature for feature, _ in samples])
        old_labels = torch.stack([label for _, label in samples])
        train_features = torch.cat((train_features, old_features))
        train_labels = torch.cat((train_labels, old_labels))

    n_epochs = int(context.run_config['local-epochs'])
    optimizer, scheduler = model.get_optimizer(total_samples * n_epochs)
    n_batches = int(context.run_config['batch-size'])
    data = TensorDataset(train_features, train_labels)
    batches = DataLoader(data, batch_size=n_batches, shuffle=True)
    model.train()

    criterion = BCEWithLogitsLoss().to(device)
    running_loss = 0.0
    for _ in range(n_epochs):
        for batch in batches:
            batch_features, batch_labels = batch
            
            # Move inputs/labels to the correct device and dtypes
            batch_features: Tensor = batch_features.to(device)
            batch_labels: Tensor = batch_labels.to(device)
            optimizer.zero_grad() # Reset gradient

            outputs: Tensor = model(batch_features) # Forward pass
            outputs = outputs.squeeze(1)
            loss: Tensor = criterion(outputs, batch_labels)
            loss.backward() # Computes mini-batch SGD

            optimizer.step() # Update weights using gradient descent
            scheduler.step() # Adjust learning rate
            running_loss += loss.item() * batch_labels.size(0)

    global total_flows
    total_flows += total_samples
    update_replay_buffer(train_set, replay_ratio)

    avg_trainloss = running_loss / total_samples
    return avg_trainloss

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_server_model(msg, context, device)

    flow_ids: list[int] = msg.content['config']['flows']
    uavids_path = "fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv"
    data = load_pd_csv(uavids_path)
    _, test_set = split_uavids(data, flow_ids)
    metrics = client_evaluate(context, model, test_set, device)
    metrics['num-examples'] = len(test_set)
    metric_record = MetricRecord(metrics)
    content = RecordDict({'metrics': metric_record})
    return Message(content=content, reply_to=msg)

def client_evaluate(
        context: Context, model: MLP, test_set: tuple[Tensor, Tensor], 
        device: torch.device) -> dict[str, float]:
    model.to(device)
    model.eval()
    loss_evaluator = BCEWithLogitsLoss().to(device)
    correct = 0
    total_loss = 0.0
    total_samples = len(test_set[1])
    all_predictions, all_probabilities, all_labels = [], [], []
    n_batches = int(context.run_config['batch-size'])
    data = TensorDataset(*test_set)
    batches = DataLoader(dataset=data, batch_size=n_batches, shuffle=False)
    with torch.no_grad():
        for batch in batches:
            features, labels = batch
            features: Tensor = features.to(device)
            labels: Tensor = labels.to(device)
            outputs: Tensor = model(features)
            outputs = outputs.squeeze(1)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= 0.5).long()
            
            correct += (predictions == labels).sum().item()
            batch_loss: Tensor = loss_evaluator(outputs, labels.float())
            total_loss += batch_loss.item() * labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    metrics = {
        'accuracy': correct / total_samples,
        'loss': total_loss / total_samples,
        'roc-auc': roc_auc(all_labels, all_probabilities),
        'pr-auc': pr_auc(all_labels, all_probabilities),
        'macro-f1': macro_f1(all_labels, all_predictions),
        'recall@fpr=1%': recall_at_fpr(all_labels, all_probabilities, 0.01)
    }
    return metrics

def split_uavids(df: pd.DataFrame, flows: list[int]) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    filtered = df.loc[flows]
    features, labels = filtered.drop('label', axis=1), filtered['label']
    features_np = features.to_numpy('float32')
    labels_np = labels.to_numpy('uint8')
    
    train_ratio = 0.8
    splits = train_test_split(
        features_np, labels_np,
        train_size=train_ratio,
        stratify=labels_np,
        # Make sure train & test same seed for train and evaluation.
        # Risks data leaking if not set to a value both use
        random_state=42
    )
    train_feat, test_feat, train_labels, test_labels = splits
    # .bool() to binarize and .float() since output is a float [0,1]
    # Assuming label 0 is normal/benign traffic
    train_set = (torch.from_numpy(train_feat),
                 torch.from_numpy(train_labels).bool().float())
    test_set = (torch.from_numpy(test_feat),
                torch.from_numpy(test_labels).bool().float())
    return train_set, test_set