from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fed_cl_ids.fed.replaybuffer import ReplayBuffer
from fed_cl_ids.models.losses import Losses
from fed_cl_ids.models.mlp import MLP

from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from torch import Tensor
import torch

from typing import Callable, Optional
from pprint import pprint
from time import time
import pandas as pd
import numpy as np
import warnings
import random
import os

# Privacy Engine
warnings.filterwarnings("ignore")

class Client:
    def __init__(self, context: Context) -> None:
        self.model: MLP = self._initialize_model(context)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.epochs = int(context.run_config['epochs'])
        self.dp_enabled = bool(context.run_config['dp-enabled'])
        self.cl_enabled = bool(context.run_config['cl-enabled'])
        self.ewc_lambda = float(context.run_config['ewc-lambda'])
        self.er_sample_rate = float(context.run_config['er-memory'])
        self.er_mix_ratio = float(context.run_config['er-alpha'])
        self.batch_size = int(context.run_config['batch-size'])
        
        self.context = context

        # Data cache
        self.dataframe: pd.DataFrame
        self.dataframe_path: str = ''

        # Continual Learning (experience replay + elastic weight consolidation)
        self.total_flows: int = 0
        n_features = int(context.run_config['n-features'])
        self.replay_buffer = ReplayBuffer(os.getpid(), n_features)
        self.prev_parameters: dict[str, Tensor] = {}
        self.fisher_diagonal: dict[str, Tensor] = {}

        # Differential privacy attributes
        self.stored_epsilon: Optional[float] = None # Return with evaluate
        self.privacy_engine = PrivacyEngine(accountant='rdp')
    
    def _initialize_model(self, context: Context) -> MLP:
        widths = str(context.run_config['mlp-widths'])
        model = MLP(
            n_features=int(context.run_config['n-features']),
            hidden_widths=[int(x) for x in widths.split(',')],
            dropout=float(context.run_config['mlp-dropout']),
            weight_decay=float(context.run_config['mlp-weight-decay']),
            lr_max=float(context.run_config['mlp-lr-max']),
            lr_min=float(context.run_config['mlp-lr-min'])
        )
        return model

    def update_model(self, parameters: dict[str, Tensor]) -> None:
        self.model.load_state_dict(parameters)
    
    def set_dataframe(self, csv_path: str) -> None:
        if self.dataframe_path and self.dataframe_path == csv_path:
            return
        self.dataframe = pd.read_csv(csv_path, index_col='FlowID')
        self.dataframe_path = csv_path

    def get_flow_data(self, flow_ids: list[int]) -> tuple[Tensor, Tensor]:
        if not hasattr(self, 'dataframe'):
            raise ValueError("Dataframe was not initialized")
        filtered = self.dataframe.loc[flow_ids]
        features, labels = filtered.drop('label', axis=1), filtered['label']
        np_features = features.to_numpy('float32')
        np_labels = labels.to_numpy('uint8')
        
        # bool() to binarize and float() since BCELoss is a probability [0,1]
        # This is assuming label 0 is normal/benign traffic
        tensor_features = torch.from_numpy(np_features)
        tensor_labels = torch.from_numpy(np_labels).bool().float()
        return tensor_features, tensor_labels

    def train(self, train_set: tuple[Tensor, Tensor]) -> float:
        self.model.train()
        train_features, train_labels = train_set
        n_new_samples = len(train_labels)

        if self.cl_enabled:
            # Could also do: (new_samples * 0.2) / 0.8 = 20%
            # if er_mix_ratio = 0.2
            true_ratio = (1 - self.er_mix_ratio) / self.er_mix_ratio
            ideal_sample_size = int(n_new_samples / true_ratio)
            
            # Ratio will be off until replay_buffer size >= replay_sample_size
            if self.replay_buffer and ideal_sample_size:
                actual_samples = min(len(self.replay_buffer), ideal_sample_size)
                samples = self.replay_buffer.sample(actual_samples)
                old_features, old_labels = samples
                train_features = torch.cat((train_features, old_features))
                train_labels = torch.cat((train_labels, old_labels))

        data = TensorDataset(train_features, train_labels)
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        model = self.model
        # Scheduler uses per batch, not per sample
        iterations = len(data_loader) * self.epochs 
        optimizer, scheduler = self.model.get_optimizers(iterations)
        running_loss = 0.0

        # Wrappers around the original objects
        if self.dp_enabled:
            model, optimizer, *_, data_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=1.0, 
                max_grad_norm=1.0, # Clipping
                clipping='flat', # Per individual sample (slow?)
                poisson_sampling=True # DP sampling method
            )
            
        start = time()
        for _ in range(self.epochs):
            for batch in data_loader:
                if not len(batch):
                    print("Empty batch")
                    continue

                batch_features, batch_labels = batch
                outputs: Tensor = model(batch_features) # Forward pass
                outputs = outputs.squeeze(1)
                loss: Tensor = self.criterion(outputs, batch_labels)

                if self.cl_enabled and self.prev_parameters:
                    ewc_loss = 0.0
                    # penalty = (λ/2) * Σ F_i(θ_i - θ*_i)²
                    for name, parameter in model.named_parameters():
                        # Private model has different prev_params
                        if name not in self.prev_parameters:
                            continue
                        f_i = self.fisher_diagonal[name]
                        nested_term = parameter - self.prev_parameters[name]
                        nested_term = nested_term.pow(2)
                        penalty = f_i * nested_term
                        ewc_loss += penalty.sum()
                    loss += (self.ewc_lambda / 2) * ewc_loss
                
                loss.backward() # Calculate where to step using mini-batch SGD
                # Step forward
                optimizer.step() # Update weights using gradient descent
                scheduler.step() # Adjust learning rate
                optimizer.zero_grad() # Reset for next batch
                running_loss += loss.item() * batch_labels.size(0)
        print(time() - start)

        if self.dp_enabled:
            delta = 1e-5
            self.stored_epsilon = self.privacy_engine.get_epsilon(delta)

        total_samples = max(1, len(train_labels))
        avg_loss = running_loss / total_samples
        if not self.cl_enabled:
            return avg_loss

        self.total_flows += n_new_samples
        sampler = tuple((feature, label) 
                        for feature, label in zip(*train_set)
                        if random.random() <= self.er_sample_rate)
        if sampler:
            new_features, new_labels = zip(*sampler)
            new_features = torch.stack(new_features)
            new_labels = torch.stack(new_labels)
            self.replay_buffer.append(new_features, new_labels)
        
        # Model has privacy wrapper around it and fisher_information
        # Find direct access to MLP module.
        if self.dp_enabled:
            self._initialize_model(self.context)
            self.update_model(model._module.state_dict())

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
        self.model.eval()
        test_features, test_labels = test_set
        data = TensorDataset(test_features, test_labels)
        batches = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        
        all_predictions, all_probabilities =  [], []
        correct = 0
        total_loss = 0.0
        
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

'''Client object needs to be cached or else each round would have to create
a new client. Context allows persistence for the process to obtain client'''
def assign_context(function: Callable) -> Callable:
    def wrapper(msg: Message, context: Context) -> Message:
        if not hasattr(context, '_client'):
            context._client = Client(context)
        return function(context._client, msg)
    return wrapper

app = ClientApp()

@app.train()
@assign_context
def client_train(client: Client, msg: Message) -> Message:
    new_parameters = msg.content['arrays'].to_torch_state_dict()
    client.update_model(new_parameters)
    client.set_dataframe("fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv")
    flow_ids: list[int] = msg.content['config']['flows']
    train_set = client.get_flow_data(flow_ids)

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
    new_parameters = msg.content['arrays'].to_torch_state_dict()
    client.update_model(new_parameters)
    client.set_dataframe("fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv")
    flow_ids: list[int] = msg.content['config']['flows']
    test_set = client.get_flow_data(flow_ids)

    metrics = client.evaluate(test_set)
    metrics['num-examples'] = len(test_set[1])
    metric_record = MetricRecord(metrics)
    content = RecordDict({'metrics': metric_record})
    return Message(content=content, reply_to=msg)