"""
Representation of a client/supernode in a federated framework.
Performs both training and evaluation with a local pytorch MLP module that gets
its parameters from the server module.
"""
from typing import Callable, Optional, cast
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import time
import os

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.common.typing import MetricRecordValues
from flwr.clientapp import ClientApp

from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers.optimizer import DPOptimizer
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import torch
import pandas as pd

from fed.differential_privacy import DifferentialPrivacy
from fed.continual_learning import ContinualLearning
from models.mlp import MLP, Adam, CosineAnnealingLR
from models.fed_metrics import FedMetrics

MAIN_PATH = Path().cwd()
DATAFRAME_PATH = MAIN_PATH / 'datasets' / 'UAVIDS-2025 Preprocessed.csv'
RUNTIME_PATH = MAIN_PATH / 'runtime'
TRACE_PATH = RUNTIME_PATH / 'trace.txt'

@dataclass()
class ClientConfiguration:
    """Stores configurations from pyproject.toml"""
    def __init__(self, context: Context) -> None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        self.epochs = cast(int, context.run_config['epochs'])
        self.batch_size = cast(int, context.run_config['batch-size'])

        # Experience replay
        self.cl_enabled = cast(bool, context.run_config['cl-enabled'])
        self.er_sample_rate = cast(float, context.run_config['er-memory'])
        self.er_mix_ratio = cast(float, context.run_config['er-mix-ratio'])
        self.ewc_lambda = cast(float, context.run_config['ewc-lambda'])

        # Differential privacy
        self.dp_enabled = cast(bool, context.run_config['dp-enabled'])
        self.clipping = cast(float, context.run_config['clipping'])
        self.noise = cast(float, context.run_config['noise'])
        self.delta = cast(float, context.run_config['delta'])

class Client:
    """Acts as an interactive instance of a client."""
    def __init__(self, context: Context) -> None:
        self.config = ClientConfiguration(context)
        self.model: MLP = self._initialize_model(context)
        self.criterion = torch.nn.BCEWithLogitsLoss() # Returns on device

        self.cl = ContinualLearning(
            er_filepath_identifier=os.getpid(), # or Node ID
            er_runtime_directory=RUNTIME_PATH,
            device=self.config.device
        )
        self.dp = DifferentialPrivacy()
        # Only training nodes can fetch epsilon.
        self.stored_epsilon: Optional[float] = None

        # Data cache (probably will be removed with CIC-IDS)
        self.dataframe: pd.DataFrame
        self.dataframe_path: Optional[Path] = None

    def _initialize_model(self, context: Context) -> MLP:
        widths = cast(str, context.run_config['mlp-widths'])
        model = MLP(
            n_features=cast(int, context.run_config['n-features']),
            hidden_widths=map(int, widths.split(',')),
            dropout=cast(float, context.run_config['mlp-dropout']),
            weight_decay=cast(float, context.run_config['mlp-weight-decay']),
            lr_max=cast(float, context.run_config['mlp-lr-max']),
            lr_min=cast(float, context.run_config['mlp-lr-min'])
        )
        return model.to(self.config.device)

    def update_model(self, parameters: dict[str, Tensor]) -> None:
        """
        Load a given MLP module's parameters into client's model.
        Automatically moves parameters to device the MLP model is on.
        Equivalent to using mlp.load_state_dict().
        """
        self.model.load_state_dict(parameters)

    def set_dataframe(self, csv_path: Path) -> None:
        """Load and cache self.dataframe for faster reuse if same csv file."""
        if self.dataframe_path is None or self.dataframe_path != csv_path:
            self.dataframe = pd.read_csv(csv_path, index_col='FlowID')
            self.dataframe_path = csv_path

    def get_flow_data(self, flow_ids: list[int]) -> tuple[Tensor, Tensor]:
        """
        Uses self.dataframe to transform table of features and label into
        tensors on self.config.device.

        Assumes label column is called 'label'.
        
        Binarizes multiclass labels. So preprocessed benign/normal traffic 
        should always be zero and malicious traffic should be a non-zero 
        integer label.
        """
        if not hasattr(self, 'dataframe'):
            raise ValueError("Dataframe was not initialized")
        filtered = self.dataframe.loc[flow_ids]
        features, labels = filtered.drop('label', axis=1), filtered['label']
        np_features = features.to_numpy('float32')
        np_labels = labels.to_numpy(bool).astype('float32')

        tensor_features = torch.from_numpy(np_features).to(self.config.device)
        tensor_labels = torch.from_numpy(np_labels).to(self.config.device)
        return tensor_features, tensor_labels

    def train(self, train_set: tuple[Tensor, Tensor], profile_on) -> tuple[float, int]:
        """
        Trains local model modified by the toml configuration file 
        (such as continual learning and differential privacy), 
        updates model, and calculates the average loss.
        """
        profile_on = False
        if profile_on:
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            with profile(activities=activities, profile_memory=True) as prof:
                result = self._train(train_set)
            TRACE_PATH.write_text(
                prof.key_averages().table(sort_by='cpu_time_total')
            )
        else:
            result = self._train(train_set)
        return result

    def _train(self, train_set: tuple[Tensor, Tensor]) -> tuple[float, int]:
        self.model.train()
        if self.config.cl_enabled:
            train_set = self.cl.er.sample_replay_buffer(
                original_dataset=train_set,
                n_new_samples=len(train_set[1]),
                ratio_old_samples=self.config.er_mix_ratio
            )

        data_loader = DataLoader(
            dataset=TensorDataset(*train_set),
            batch_size=self.config.batch_size,
            shuffle=True
        )
        data_loader = cast(DataLoader[tuple[Tensor, Tensor]], data_loader)
        # PrivacyEngine objects wrap around the original objects
        # N forward passes + N backward passes per batch (where N = batch size)
        # Time is relative to size of dataset to train
        packages = self._create_model_packages(data_loader)
        training_model, optimizer, data_loader = packages
        scheduler = self.model.get_scheduler(
            optimizer=optimizer,
            cosine_epochs=len(data_loader) * self.config.epochs
        )

        # Total_loss is mutated in _train_iteration()
        total_loss = self._zero_tensor()
        total_samples = 0
        loop_start = time()
        for _ in range(self.config.epochs):
            loss, n_samples = self._train_iteration(
                model=training_model,
                optimizer=optimizer,
                scheduler=scheduler,
                data_loader=data_loader
            )
            total_loss += loss
            total_samples += n_samples
        print(f"Training Loop: {time() - loop_start} sec")

        if isinstance(training_model, GradSampleModule):
            self.model = cast(MLP, training_model.to_standard_module())
            if self.stored_epsilon is not None:
                raise TypeError("Epsilon needs to be None from evaluation.")
            self.stored_epsilon = self.dp.get_epsilon(self.config.delta)

        if self.config.cl_enabled:
            self.cl.er.add_data(
                original_dataset=train_set,
                sample_rate=self.config.er_sample_rate
            )
            self.cl.ewc.update_fisher_information(
                model=self.model,
                train_set=train_set,
                criterion=self.criterion,
                batch_size=self.config.batch_size
            )
            self.cl.ewc.update_prev_parameters(self.model)
        average_loss = total_loss / max(1, total_samples)
        return average_loss.item(), total_samples // self.config.epochs

    def _train_iteration(
            self,
            model: MLP | GradSampleModule,
            optimizer: Adam | DPOptimizer,
            scheduler: CosineAnnealingLR,
            data_loader: DataLoader[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, int]:
        n_samples: int = 0
        iteration_loss = self._zero_tensor()
        for batch in data_loader:
            batch_features, batch_labels = cast(tuple[Tensor, Tensor], batch)
            # Poisson sampling could produce empty batch
            if not batch_labels.nelement():
                continue

            outputs: Tensor = model(batch_features) # Forward pass
            outputs = outputs.squeeze(1)
            loss: Tensor = self.criterion(outputs, batch_labels)

            # As the model changes from its gradients,
            # calculate penalties as it changes parameters
            if self.config.cl_enabled and not self.cl.ewc.is_empty():
                batch_fisher_penalty = self.cl.ewc.calculate_penalty(
                    model=model,
                    ewc_lambda=self.config.ewc_lambda
                )
                loss += batch_fisher_penalty

            optimizer.zero_grad() # Prevents gradient accumulation
            loss.backward() # Calculate where to step using mini-batch SGD
            optimizer.step() # Step forward down gradient
            scheduler.step() # Adjusts learning rate

            batch_size = len(batch_labels)
            n_samples += batch_size
            iteration_loss += loss.detach() * batch_size
        return iteration_loss, n_samples

    @torch.no_grad()
    def evaluate(self, test_set: tuple[Tensor, Tensor]) -> tuple[dict[str, float], int]:
        """Evaluates server aggregated model against the test set."""
        self.model.eval()
        data_loader = DataLoader(
            dataset=TensorDataset(*test_set),
            batch_size=self.config.batch_size
        )
        data_loader = cast(DataLoader[tuple[Tensor, Tensor]], data_loader)
        packages = self._create_model_packages(data_loader)
        evaluation_model, _, data_loader = packages

        labels_list: list[Tensor] =  []
        predictions_list: list[Tensor] = []
        probabilities_list: list[Tensor] = []
        n_correct = self._zero_tensor()
        total_loss = self._zero_tensor()
        total_samples = 0

        for batch in data_loader:
            batch_features, batch_labels = cast(tuple[Tensor, Tensor], batch)
            if not batch_labels.nelement():
                continue

            outputs: Tensor = evaluation_model(batch_features)
            outputs = outputs.squeeze(1)
            batch_probabilities = torch.sigmoid(outputs)
            batch_predictions = (batch_probabilities >= 0.5).float()

            n_batch_correct = (batch_predictions == batch_labels).sum()
            n_correct += n_batch_correct
            batch_loss: Tensor = self.criterion(outputs, batch_labels)
            total_loss += batch_loss * len(batch_labels)
            total_samples += len(batch_labels)

            labels_list.append(batch_labels)
            predictions_list.append(batch_predictions)
            probabilities_list.append(batch_probabilities)

        if isinstance(evaluation_model, GradSampleModule):
            self.model = cast(MLP, evaluation_model.to_standard_module())

        metrics = {
            'accuracy': (n_correct / total_samples).item(),
            'loss': (total_loss / total_samples).item(),
        }
        fed_metrics = self._create_metrics(
            labels=torch.cat(labels_list).cpu(),
            predictions=torch.cat(predictions_list).cpu(),
            probabilities=torch.cat(probabilities_list).cpu()
        )
        metrics.update(fed_metrics)
        return metrics, total_samples

    def _create_metrics(
            self,
            labels: Tensor,
            predictions: Tensor,
            probabilities: Tensor
    ) -> dict[str, float]:
        """Generates general metrics and resets the epsilon it has value."""
        training_epsilon = -1 # Default sentinel value
        if self.stored_epsilon is not None:
            training_epsilon = self.stored_epsilon
            self.stored_epsilon = None
        fed_cl_ids_metrics = {
            'auroc': FedMetrics.auroc(labels, probabilities),
            'auprc': FedMetrics.auprc(labels, probabilities),
            'macro-f1': FedMetrics.macro_f1(labels, predictions),
            'recall@fpr=1%': FedMetrics.recall_at_fpr(labels, probabilities, 0.01),
            'epsilon': training_epsilon
        }
        return fed_cl_ids_metrics

    def _create_model_packages(
            self,
            data_loader: DataLoader[tuple[Tensor, Tensor]]
    ) -> tuple[MLP | GradSampleModule, Adam | DPOptimizer, DataLoader[tuple[Tensor, Tensor]]]:
        """
        Return new private wrappers around original objects if
        differential privacy is enabled.
        """
        if not self.config.dp_enabled:
            return self.model, self.model.get_optimizer(), data_loader
        if is_evaluating := not self.model.training:
            self.model.train()
        dp_model, dp_optimizer, *_, dp_data_loader = self.dp.make_private(
            module=self.model,
            optimizer=self.model.get_optimizer(),
            data_loader=data_loader,
            noise_multiplier=self.config.noise,
            max_grad_norm=self.config.clipping # Clipping value
        )
        if is_evaluating:
            dp_model.eval()
        return dp_model, dp_optimizer, dp_data_loader

    def _zero_tensor(self) -> Tensor:
        """
        Mainly used to avoid .item() sync overhead for tensors since
        you can queue tensor operations for GPU to do.
        """
        return torch.tensor(0.0, dtype=torch.float32, device=self.config.device)


def _assign_client(function: Callable) -> Callable:
    """Initialize and cache client to context to maintain persistence."""
    def wrapper(msg: Message, context: Context) -> Message:
        context.client = getattr(context, 'client', Client(context))
        return function(context.client, msg)
    return wrapper

app = ClientApp()

@app.train()
@_assign_client
def client_train(client: Client, msg: Message) -> Message:
    """
    Initializes dataframe for client, trains, and constructs message to send
    back to server.
    """
    profile_on = 'profile_on' in msg.content['config']
    server_arrays = cast(ArrayRecord, msg.content['arrays'])
    server_parameters = server_arrays.to_torch_state_dict()
    client.update_model(server_parameters)
    client.set_dataframe(DATAFRAME_PATH)
    flow_ids = cast(list[int], msg.content['config']['flows'])
    train_set = client.get_flow_data(flow_ids)
    average_loss, n_samples = client.train(train_set, profile_on)
    # state_dict() auto detaches from grad, but still on cuda
    client_parameters = cast(OrderedDict, client.model.state_dict())
    metrics = {
        'train_loss': average_loss,
        'num-examples': n_samples
    }
    content = RecordDict({
        'arrays': ArrayRecord(client_parameters),
        'metrics': MetricRecord(metrics)
    })
    return Message(content, reply_to=msg)

@app.evaluate()
@_assign_client
def client_evaluate(client: Client, msg: Message) -> Message:
    """
    Initializes dataframe for client, evaluates, and constructs message to send
    back to server. 
    """
    server_arrays = cast(ArrayRecord, msg.content['arrays'])
    server_parameters = server_arrays.to_torch_state_dict()
    client.update_model(server_parameters)
    client.set_dataframe(DATAFRAME_PATH)
    flow_ids: list[int] = cast(list[int], msg.content['config']['flows'])
    test_set = client.get_flow_data(flow_ids)

    eval_metrics, n_samples = cast(
        tuple[dict[str, MetricRecordValues], int],
        client.evaluate(test_set)
    )
    eval_metrics['num-examples'] = n_samples
    content = RecordDict({'metrics': MetricRecord(eval_metrics)})
    return Message(content, reply_to=msg)
