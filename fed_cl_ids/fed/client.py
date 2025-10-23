from fed_cl_ids.models.losses import roc_auc, pr_auc, macro_f1, recall_at_fpr
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from fed_cl_ids.models.mlp import MLP
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn import CrossEntropyLoss
from torch import Tensor
import torch
import pandas as pd
import numpy as np

# Flower ClientApp
app = ClientApp()

dataset: pd.DataFrame | None = None

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

# Client MLP model needs to receive initial parameters to construct model.
@app.train()
def train(msg: Message, context: Context) -> Message:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_server_model(msg, context, device)
    
    flow_ids: list[int] = msg.content['config']['flows']
    uavids_path = "fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv"
    global dataset
    if dataset is None:
        dataset = pd.read_csv(uavids_path).set_index('FlowID')
    batch_size = int(context.run_config['batch-size'])
    train, _ = split_uavids(dataset, flow_ids, batch_size)
    epochs = int(context.run_config['local-epochs'])
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

def client_train(
        model: MLP, train_data: DataLoader, 
        n_epochs: int, device: torch.device) -> float:
    criterion = CrossEntropyLoss().to(device)
    optimizer, scheduler = model.get_optimizer(len(train_data) * 20)
    model.train()
    running_loss = 0.0
    for _ in range(n_epochs):
        for batch in train_data:
            features, labels = batch
            # Move inputs/labels to the correct device and dtypes
            features = features.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.long)
            optimizer.zero_grad() # Resets the gradient

            outputs = model(features) # Forward pass
            loss = criterion(outputs, labels)
            loss.backward() # Computes mini-batch SGD

            optimizer.step() # Update weights using gradient descent
            scheduler.step() # Adjust learning rate
            running_loss += loss.item()
    avg_trainloss = running_loss / len(train_data)
    return avg_trainloss

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_server_model(msg, context, device)

    flow_ids: list[int] = msg.content['config']['flows']
    uavids_path = "fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv"
    global dataset
    if dataset is None:
        dataset =  pd.read_csv(uavids_path).set_index('FlowID')
    batch_size = int(context.run_config['batch-size'])
    _, test = split_uavids(dataset, flow_ids, batch_size)

    metrics = client_evaluate(model, test, device)
    metrics['num-examples'] = len(test)
    metric_record = MetricRecord(metrics)
    content = RecordDict({'metrics': metric_record})
    return Message(content=content, reply_to=msg)

def client_evaluate(model: MLP, test_data: DataLoader, 
                    device: torch.device) -> dict[str, float]:
    model.to(device)
    loss_evaluator = CrossEntropyLoss().to(device)
    correct = total_samples = 0
    total_loss = 0.0
    all_predictions, all_probabilities, all_labels = [], [], []
    # Doesn't need gradient descent for evaluation
    with torch.no_grad():
        for batch in test_data:
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
        'accuracy': correct / total_samples if total_samples > 0 else 0.0,
        'loss': total_loss / len(test_data),
        'roc-auc': roc_auc(all_labels, all_predictions),
        'pr-auc': pr_auc(all_labels, all_predictions),
        'macro-f1': macro_f1(all_labels, all_predictions),
        'recall@fpr=1%': recall_at_fpr(all_labels, all_probabilities, 0.01)
    }
    return metrics

# Get flow data from dataframe, split data into train and test
def split_uavids(df: pd.DataFrame, flows: list[int], n_batches: int):
    filtered = df.loc[flows]
    features, labels = filtered.drop('label', axis=1), filtered['label']
    features_tensor = torch.from_numpy(features.to_numpy())
    labels_tensor = torch.from_numpy(labels.to_numpy())
    dataset = TensorDataset(features_tensor, labels_tensor)

    train_ratio, test_ratio = (0.8, 0.2)
    train_set, test_set = random_split(dataset, (train_ratio, test_ratio),
                                       torch.Generator().manual_seed(0))
    train = DataLoader(train_set, batch_size=n_batches, shuffle=True)
    test = DataLoader(test_set, batch_size=n_batches, shuffle=True)
    return train, test