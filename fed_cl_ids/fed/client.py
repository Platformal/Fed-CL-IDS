from fed_cl_ids.models.losses import roc_auc, pr_auc, macro_f1, recall_at_fpr
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from sklearn.model_selection import train_test_split
from flwr.clientapp import ClientApp
from fed_cl_ids.models.mlp import MLP
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
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
    """Randomly add samples to replay buffer, maintaining max size."""
    # Total flows should be updated before training
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
    replay_ratio = float(context.run_config['replay-ratio'])    
    flow_ids: list[int] = msg.content['config']['flows']

    global total_flows
    total_flows += len(flow_ids)

    uavids_path = "fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv"
    data = load_pd_csv(uavids_path)
    batch_size = int(context.run_config['batch-size'])
    train_set, _ = split_uavids(data, flow_ids, batch_size)
    epochs = int(context.run_config['local-epochs'])
    avg_loss = client_train(model, train_set, epochs, replay_ratio, device)
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
        model: MLP, train_set: tuple[Tensor, Tensor], 
        n_epochs: int, replay_ratio: float, device: torch.device) -> float:
    criterion = CrossEntropyLoss().to(device)
    optimizer, scheduler = model.get_optimizer(len(train_set) * 20)
    model.train()
    running_loss = 0.0

    features, labels = train_set
    replay_size = int(replay_ratio * len(features))
    if replay_buffer and replay_size:
        samples = random.sample(
            replay_buffer, min(replay_size, len(replay_buffer)))
        old_features = torch.stack([feature for feature, _ in samples])
        old_labels = torch.stack([label for _, label in samples])
        features = torch.cat((features, old_features))
        labels = torch.cat((labels, old_labels))

    data = TensorDataset(features, labels)
    batches = DataLoader(data, batch_size=128, shuffle=True)
    for _ in range(n_epochs):
        for batch in batches:
            features, labels = batch
            
            # Move inputs/labels to the correct device and dtypes
            features = features.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.long)
            optimizer.zero_grad() # Reset gradient

            outputs = model(features) # Forward pass
            loss = criterion(outputs, labels)
            loss.backward() # Computes mini-batch SGD

            optimizer.step() # Update weights using gradient descent
            scheduler.step() # Adjust learning rate
            running_loss += loss.item()

    update_replay_buffer(train_set, replay_ratio)
            
    avg_trainloss = running_loss / len(train_set)
    return avg_trainloss

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_server_model(msg, context, device)

    flow_ids: list[int] = msg.content['config']['flows']
    uavids_path = "fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv"
    data = load_pd_csv(uavids_path)
    batch_size = int(context.run_config['batch-size'])
    _, test_set = split_uavids(data, flow_ids, batch_size)

    metrics = client_evaluate(model, test_set, device)
    metrics['num-examples'] = len(test_set)
    metric_record = MetricRecord(metrics)
    content = RecordDict({'metrics': metric_record})
    return Message(content=content, reply_to=msg)

def client_evaluate(model: MLP, test_data: tuple[Tensor, Tensor], 
                    device: torch.device) -> dict[str, float]:
    model.to(device)
    loss_evaluator = CrossEntropyLoss().to(device)
    correct = total_samples = 0
    total_loss = 0.0
    all_predictions, all_probabilities, all_labels = [], [], []
    # Doesn't need gradient descent for evaluation
    data = TensorDataset(*test_data)
    batches = DataLoader(dataset=data, batch_size=128, shuffle=True)
    with torch.no_grad():
        for batch in batches:
            features, labels = batch
            features: Tensor = features.to(device).to(torch.float32)
            labels: Tensor = labels.to(device).to(torch.long)
            outputs: Tensor = model(features)
            predictions = torch.argmax(outputs, dim=1)
            # Outputs probability for each class. Take positive probability
            probabilities = torch.softmax(outputs, dim=1)[:,1]
            
            correct += (predictions == labels).sum().item()
            total_loss += loss_evaluator(outputs, labels).item()
            total_samples += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    metrics = {
        'accuracy': correct / total_samples if total_samples else 0.0,
        'loss': total_loss / len(test_data),
        'roc-auc': roc_auc(all_labels, all_predictions),
        'pr-auc': pr_auc(all_labels, all_predictions),
        'macro-f1': macro_f1(all_labels, all_predictions),
        'recall@fpr=1%': recall_at_fpr(all_labels, all_probabilities, 0.01)
    }
    return metrics

# Get flow data from dataframe, split data into train and test
def split_uavids(df: pd.DataFrame, flows: list[int], n_batches: int) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    filtered = df.loc[flows]
    features, labels = filtered.drop('label', axis=1), filtered['label']
    features_np = features.to_numpy('float32')
    labels_np = labels.to_numpy('uint8')
    
    train_ratio = 0.8
    splits = train_test_split(
        features_np, labels_np,
        train_size=train_ratio,
        random_state=0,
        stratify=labels_np # Ensure equal distribution among train & test
    )
    train_feat, test_feat, train_labels, test_labels = splits
    # Binarize multi-classification
    train_set = (torch.from_numpy(train_feat),
                 torch.from_numpy(train_labels).bool().long())
    test_set = (torch.from_numpy(test_feat),
                torch.from_numpy(test_labels).bool().long())
    return train_set, test_set