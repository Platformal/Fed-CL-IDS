from fed_cl_ids.models.losses import Losses
from fed_cl_ids.models.mlp import MLP
from sklearn.model_selection import train_test_split
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss
from torch import Tensor
import torch
import pandas as pd
import numpy as np
import random

# Flower ClientApp
app = ClientApp()

dataset: pd.DataFrame | None = None

# Experience replay variables
replay_buffer: list[tuple[Tensor, Tensor]] = []
total_flows: int = 0

# Elastic weight consolidation variables
previous_parameters: dict[str, Tensor] | None = None
fisher_diagonal: dict[str, Tensor] | None = None

# Server sends message of array dict, initialize and load model parameters.
def new_server_model(msg: Message, context: Context, device: torch.device) -> MLP:
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

def split_uavids(df: pd.DataFrame, flows: list[int]) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    filtered = df.loc[flows]
    features, labels = filtered.drop('label', axis=1), filtered['label']
    features_np = features.to_numpy('float32')
    labels_np = labels.to_numpy('uint8')
    
    train_ratio = 0.8
    splits = train_test_split(
        features_np, labels_np,
        train_size=train_ratio,
        # Equal distribution of labels for train and evaluate
        stratify=labels_np,
        # Risks data leaking if not set to a value both train & eval use
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

def load_pd_csv(file_path) -> pd.DataFrame:
    global dataset
    if dataset is None:
        dataset = pd.read_csv(file_path).set_index('FlowID')
    return dataset

# Client MLP model needs to receive initial parameters to construct model.
@app.train()
def train(msg: Message, context: Context) -> Message:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = new_server_model(msg, context, device)
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

    continual_learning = bool(context.run_config['cl-enabled'])
    ewc_half_lambda = int(context.run_config['ewc-lambda']) / 2
    n_new_samples = len(train_labels)

    if continual_learning:
        replay_mix_ratio = float(context.run_config['er-mix'])
        true_ratio = (1 - replay_mix_ratio) / replay_mix_ratio
        replay_sample_size = int(n_new_samples/ true_ratio)

        # Ratio will be off until replay_buffer size >= replay_sample_size
        if replay_buffer and replay_sample_size:
            samples = random.sample(
                population=replay_buffer, 
                k=min(replay_sample_size, len(replay_buffer))
            )
            old_features = torch.stack([feature for feature, _ in samples])
            old_labels = torch.stack([label for _, label in samples])
            train_features = torch.cat((train_features, old_features))
            train_labels = torch.cat((train_labels, old_labels))

    total_samples = len(train_labels)
    n_epochs = int(context.run_config['local-epochs'])
    n_batches = int(context.run_config['batch-size'])
    data = TensorDataset(train_features, train_labels)
    batches = DataLoader(data, batch_size=n_batches, shuffle=True)

    optimizer, scheduler = model.get_optimizer(total_samples * n_epochs)
    criterion = BCEWithLogitsLoss().to(device)
    running_loss = 0.0
    model.train()

    for _ in range(n_epochs):
        for batch in batches:
            batch_features, batch_labels = batch
            batch_features: Tensor = batch_features.to(device)
            batch_labels: Tensor = batch_labels.to(device)

            outputs: Tensor = model(batch_features) # Forward pass
            outputs = outputs.squeeze(1)
            loss: Tensor = criterion(outputs, batch_labels)

            if continual_learning:
                global previous_parameters, fisher_diagonal
                if not (previous_parameters and fisher_diagonal):
                    break
                ewc_loss = 0.0
                # penalty = (λ/2) * Σ F_i(θ_i - θ*_i)²
                for name, parameter in model.named_parameters():
                    # Doesn't catch anything, defined by fisher function
                    if name not in previous_parameters:
                        print(f"{name} not in previous_params")
                        continue
                    f_i = fisher_diagonal[name]
                    nested_term = parameter - previous_parameters[name]
                    nested_term = nested_term.pow(2)
                    penalty = f_i * nested_term
                    ewc_loss += penalty.sum()
                loss += ewc_half_lambda * ewc_loss
            
            loss.backward() # Computes mini-batch SGD
            optimizer.step() # Update weights using gradient descent
            scheduler.step() # Adjust learning rate
            optimizer.zero_grad() # Reset for next batch
            running_loss += loss.item() * batch_labels.size(0)
    avg_loss = running_loss / total_samples
    if not continual_learning:
        return avg_loss

    global total_flows
    total_flows += n_new_samples
    replay_memory_ratio = float(context.run_config['er-memory'])
    update_replay_buffer(train_set, replay_memory_ratio)

    fisher_diagonal = fisher_information(model, train_set, n_batches, criterion)
    previous_parameters = {
        name: parameter.clone().detach()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return avg_loss

def fisher_information(model: MLP, train_set: tuple[Tensor, Tensor], n_batches: int,
                        criterion: BCEWithLogitsLoss) -> dict[str, Tensor]:
    # Doesn't need to go back to train since training is done
    model.eval()
    fisher: dict[str, Tensor] = {
        name: torch.zeros_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    data = TensorDataset(*train_set)
    batches = DataLoader(data, batch_size=n_batches, shuffle=False)
    for batch in batches:
        features, labels = batch
        model.zero_grad()

        outputs: Tensor = model(features)
        outputs = outputs.squeeze(1)
        loss: Tensor = criterion(outputs, labels)
        loss.backward()

        # Sum of square gradients
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                fisher[name] += parameter.grad.detach() ** 2
    
    # Averaging out after total sum calculated
    n_samples = len(batches)
    for name in fisher:
        fisher[name] /= n_samples

    return fisher
        
# If episodic memory = 0.05%, then it needs 200 to store 1 to replay buffer.
# Replace with random mask per element
def update_replay_buffer(train_set: tuple[Tensor, Tensor], replay_ratio: float) -> None:
    data_length = len(train_set[1])
    n_samples = int(data_length * replay_ratio)
    if not n_samples:
        return
    samples = random.sample(tuple(zip(*train_set)), n_samples)
    global replay_buffer
    replay_buffer.extend(samples)

    max_buffer_size = int(replay_ratio * total_flows)
    if len(replay_buffer) > max_buffer_size:
        replay_buffer = replay_buffer[-max_buffer_size:]

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = new_server_model(msg, context, device)
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
    model.eval()
    loss_evaluator = BCEWithLogitsLoss().to(device)
    correct = 0
    total_loss = 0.0
    total_samples = len(test_set[1])
    all_predictions, all_probabilities =  [], []
    n_batches = int(context.run_config['batch-size'])
    test_features, test_labels = test_set
    data = TensorDataset(test_features, test_labels)
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
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = test_labels.numpy()

    metrics = {
        'accuracy': correct / total_samples,
        'loss': total_loss / total_samples,
        'roc-auc': Losses.roc_auc(all_labels, all_probabilities),
        'pr-auc': Losses.pr_auc(all_labels, all_probabilities),
        'macro-f1': Losses.macro_f1(all_labels, all_predictions),
        'recall@fpr=1%': Losses.recall_at_fpr(all_labels, all_probabilities, 0.01)
    }
    return metrics