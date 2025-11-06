from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from sklearn.model_selection import train_test_split
from fed_cl_ids.models.losses import Losses
from fed_cl_ids.fed.replaybuffer import ReplayBuffer
from fed_cl_ids.models.mlp import MLP

from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import torch

from typing import Callable
from pprint import pprint
import pandas as pd
import numpy as np
import random
import os

# Flower ClientApp
app = ClientApp()

class Client:
    def __init__(self) -> None:
        self.dataframe: pd.DataFrame
        self.model: MLP
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # toml configurables
        self.epochs: int
        self.cl_enabled: bool
        self.ewc_half_lambda: float
        self.er_sample_rate: float
        self.er_priority: float
        self.batch_size: int

        # Continual learning attributes
        self.total_flows: int = int()
        self.replay_buffer: ReplayBuffer
        self.replay_features: Tensor = torch.empty((0,))
        self.replay_labels: Tensor = torch.empty((0,))
        self.prev_parameters: dict[str, Tensor] = {}
        self.fisher_diagonal: dict[str, Tensor] = {}
    
    # Remove n_classes
    def _initialize_model(self, context: Context) -> None:
        widths = str(context.run_config['mlp-widths'])
        new_model = MLP(
            n_features=int(context.run_config['n-features']),
            n_classes=int(context.run_config['n-classes']),
            hidden_widths=[int(x) for x in widths.split(',')],
            dropout=float(context.run_config['mlp-dropout']),
            weight_decay=float(context.run_config['mlp-weight-decay']),
            lr_max=float(context.run_config['mlp-lr-max']),
            lr_min=float(context.run_config['mlp-lr-min'])
        )
        self.model = new_model

    def update_model(self, msg: Message) -> None:
        new_model_params = msg.content['arrays'].to_torch_state_dict()
        self.model.load_state_dict(new_model_params)
    
    def set_dataframe(self, path_to_csv: str) -> None:
        if hasattr(self, 'dataframe'):
            return
        self.dataframe = pd.read_csv(path_to_csv).set_index('FlowID')

    def train_test_split(
            self, flow_ids: list[int], train_ratio: float = 0.8, 
            random_seed: int = 42, label_parity: bool = True
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        '''
        Split client's dataframe into train and test sets.\n
        ****
        **label_parity**: bool\n
        Whether to distribute multi-class labels evenly.
        '''
        if self.dataframe is None:
            raise ValueError("Dataframe was not initialized")
        filtered = self.dataframe.loc[flow_ids]
        features, labels = filtered.drop('label', axis=1), filtered['label']
        features_np = features.to_numpy('float32')
        labels_np = labels.to_numpy('uint8')
        
        splits = train_test_split(
            features_np, labels_np,
            train_size=train_ratio,
            stratify=labels_np if label_parity else None,
            random_state=random_seed
        )
        train_features, test_features, train_labels, test_labels = splits

        # bool() to binarize and float() since BCELoss is a probability [0,1]
        # This is assuming label 0 is normal/benign traffic
        train_labels = torch.from_numpy(train_labels).bool().float()
        test_labels = torch.from_numpy(test_labels).bool().float()
        train_features = torch.from_numpy(train_features)
        test_features = torch.from_numpy(test_features)
    
        train_set = (train_features, train_labels)
        test_set = (test_features, test_labels)
        return train_set, test_set

    def train(self, train_set: tuple[Tensor, Tensor]) -> float:
        train_features, train_labels = train_set
        n_new_samples = len(train_labels)

        if self.cl_enabled:
            true_ratio = (1 - self.er_priority) / self.er_priority
            replay_sample_size = int(n_new_samples / true_ratio)

            # Ratio will be off until replay_buffer size >= replay_sample_size
            if self.replay_buffer and replay_sample_size:
                max_samples = min(len(self.replay_buffer), replay_sample_size)
                old_features, old_labels = self.replay_buffer.sample(max_samples)
                train_features = torch.cat((train_features, old_features))
                train_labels = torch.cat((train_labels, old_labels))

        total_samples = len(train_labels)
        data = TensorDataset(train_features, train_labels)
        batches = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        # Optimizer corresponds per batch, not sample
        iterations = len(batches) * self.epochs
        optimizer, scheduler = self.model.get_optimizer(iterations)
        running_loss = 0.0
        self.model.train()

        for _ in range(self.epochs):
            for batch in batches:
                batch_features, batch_labels = batch

                outputs: Tensor = self.model(batch_features) # Forward pass
                outputs = outputs.squeeze(1)
                loss: Tensor = self.criterion(outputs, batch_labels)

                if self.cl_enabled and self.prev_parameters:
                    ewc_loss = 0.0
                    # penalty = (λ/2) * Σ F_i(θ_i - θ*_i)²
                    for name, parameter in self.model.named_parameters():
                        # Doesn't catch anything, defined by fisher function
                        if name not in self.prev_parameters:
                            print(f"{name} not in previous_params")
                            continue
                        f_i = self.fisher_diagonal[name]
                        nested_term = parameter - self.prev_parameters[name]
                        nested_term = nested_term.pow(2)
                        penalty = f_i * nested_term
                        ewc_loss += penalty.sum()
                    loss += self.ewc_half_lambda * ewc_loss
                
                loss.backward() # Computes mini-batch SGD
                optimizer.step() # Update weights using gradient descent
                scheduler.step() # Adjust learning rate
                optimizer.zero_grad() # Reset for next batch
                running_loss += loss.item() * batch_labels.size(0)
        avg_loss = running_loss / total_samples
        if not self.cl_enabled:
            return avg_loss

        self.total_flows += n_new_samples
        sampler = ((feature, label) 
                for feature, label in zip(*train_set)
                if random.random() <= self.er_sample_rate)
        if sampler:
            new_features, new_labels = zip(*sampler)
            self.replay_buffer.append(torch.stack(new_features), torch.stack(new_labels))

        self.fisher_diagonal = self._fisher_information(train_set)
        self.prev_parameters = {
            name: parameter.clone().detach()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        return avg_loss
    
    def _fisher_information(self, train_set: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        self.model.eval()
        fisher: dict[str, Tensor] = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        data = TensorDataset(*train_set)
        batches = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        for batch in batches:
            features, labels = batch
            self.model.zero_grad()
            outputs: Tensor = self.model(features)
            outputs = outputs.squeeze(1)
            loss: Tensor = self.criterion(outputs, labels)
            loss.backward()

            # Sum of square gradients
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is not None:
                        fisher[name] += parameter.grad.detach().pow(2)
        
        # Averaging out after total sum calculated
        n_samples = len(batches)
        for name in fisher:
            fisher[name] /= n_samples
        return fisher
    
    def evaluate(self, test_set: tuple[Tensor, Tensor]) -> dict[str, float]:
        test_features, test_labels = test_set
        data = TensorDataset(test_features, test_labels)
        batches = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        
        all_predictions, all_probabilities =  [], []
        correct = 0
        total_loss = 0.0
        self.model.eval()
        
        with torch.no_grad():
            for batch in batches:
                features, labels = batch
                features: Tensor
                labels: Tensor
                outputs: Tensor = self.model(features)
                outputs = outputs.squeeze(1)
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities >= 0.5).long()
                
                correct += (predictions == labels).sum().item()
                batch_loss: Tensor = self.criterion(outputs, labels.float())
                total_loss += batch_loss.item() * labels.size(0)

                all_predictions.extend(predictions.numpy())
                all_probabilities.extend(probabilities.numpy())
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = test_labels.numpy()
        total_samples = len(test_set[1])

        metrics = {
            'accuracy': correct / total_samples,
            'loss': total_loss / total_samples,
            'roc-auc': Losses.roc_auc(all_labels, all_probabilities),
            'pr-auc': Losses.pr_auc(all_labels, all_probabilities),
            'macro-f1': Losses.macro_f1(all_labels, all_predictions),
            'recall@fpr=1%': Losses.recall_at_fpr(all_labels, all_probabilities, 0.01)
        }
        return metrics
    
def assign_context(function: Callable):
    def wrapper(msg: Message, context: Context) -> Message:
        client = Client()
        client._initialize_model(context)
        client.epochs = int(context.run_config['epochs'])
        client.cl_enabled = bool(context.run_config['cl-enabled'])
        client.ewc_half_lambda = float(context.run_config['ewc-lambda']) / 2
        client.er_sample_rate = float(context.run_config['er-memory'])
        client.er_priority = float(context.run_config['er-alpha'])
        client.batch_size = int(context.run_config['batch-size'])
        n_features = int(context.run_config['n-features'])
        client.replay_buffer = ReplayBuffer(os.getpid(), n_features)
        
        if function.__name__ == "client_train" and hasattr(context, '_cl'):
            client.total_flows =  context._cl['total_flows']
            client.replay_buffer = context._cl['replay_buffer']
            client.prev_parameters = context._cl['prev_parameters']
            client.fisher_diagonal = context._cl['fisher_diagonal']

        client_results: Message = function(client, msg)

        if function.__name__ == "client_train":
            context._cl = {
            'total_flows': client.total_flows,
            'replay_buffer': client.replay_buffer,
            'prev_parameters': client.prev_parameters,
            'fisher_diagonal': client.fisher_diagonal
        }
        return client_results
    return wrapper

@app.train()
@assign_context
def client_train(client: Client, msg: Message) -> Message:
    client.update_model(msg)
    client.set_dataframe("fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv")
    flow_ids: list[int] = msg.content['config']['flows']
    train_set, _ = client.train_test_split(flow_ids)

    average_loss = client.train(train_set)
    client_model_params = ArrayRecord(client.model.state_dict())
    metrics = {
        'train_loss': average_loss,
        'num-examples': len(train_set[1])
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({
        'arrays': client_model_params, 
        'metrics': metric_record
    })
    return Message(content, reply_to=msg)

@app.evaluate()
@assign_context
def client_evaluate(client: Client, msg: Message) -> Message:
    client.update_model(msg)
    client.set_dataframe("fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv")
    flow_ids: list[int] = msg.content['config']['flows']
    _, test_set = client.train_test_split(flow_ids)

    metrics = client.evaluate(test_set)
    metrics['num-examples'] = len(test_set[1])
    metric_record = MetricRecord(metrics)
    content = RecordDict({'metrics': metric_record})
    return Message(content=content, reply_to=msg)