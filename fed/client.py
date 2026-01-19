"""
Representation of a client/supernode in a federated framework.
Performs both training and evaluation with a local pytorch MLP module that gets
its parameters from the server module.
"""
from typing import Callable, Iterable, cast
from collections import OrderedDict
from pathlib import Path
from time import time
import os

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.common.typing import MetricRecordValues
from flwr.clientapp import ClientApp

from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers.optimizer import DPOptimizer
from sklearn.preprocessing import RobustScaler
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import torch
import pandas as pd

from fed.differential_privacy import DifferentialPrivacy
from fed.continual_learning import ContinualLearning
from models.mlp import MLP, Adam, CosineAnnealingLR
from models.fed_metrics import FedMetrics

CWD_PATH = Path().cwd()
RUNTIME_PATH = CWD_PATH / 'runtime'
TRACE_PATH = RUNTIME_PATH / 'trace.txt'

class Client:
    """Acts as an interactive instance of a client."""

    scaler = RobustScaler()
    criterion = torch.nn.BCEWithLogitsLoss() # Returns tensor on device

    def __init__(self, context: Context) -> None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        self.epochs = cast(int, context.run_config['epochs'])
        self.batch_size = cast(int, context.run_config['batch-size'])
        self.mu = cast(float, context.run_config['proximal-mu'])
        self.model: MLP = self._initialize_model(context)

        self.dp = DifferentialPrivacy()
        self.dp_enabled = cast(bool, context.run_config['dp-enabled'])
        self.clipping = cast(float, context.run_config['clipping'])
        self.noise = cast(float, context.run_config['noise'])
        self.delta = cast(float, context.run_config['delta'])

        self.cl = ContinualLearning(
            er_filepath_identifier=os.getpid(), # or Node ID
            er_runtime_directory=RUNTIME_PATH,
            device=self.device
        )
        self.cl_enabled = cast(bool, context.run_config['cl-enabled'])
        self.er_sample_rate = cast(float, context.run_config['er-memory'])
        self.er_mix_ratio = cast(float, context.run_config['er-mix-ratio'])
        self.ewc_lambda = cast(float, context.run_config['ewc-lambda'])

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
        return model.to(self.device)

    def update_model(self, parameters: dict[str, Tensor]) -> None:
        """Load torch_state_dict into client model and self.server_model."""
        self.model.load_state_dict(parameters)

    def data_from_indices(
            self,
            filepath: Path,
            indices: list[int]
    ) -> tuple[Tensor, Tensor]:
        """
        Reads from given filepath and returns a tuple of containing the
        locally scaled features and the labels binarized, e.g. float(bool(label))
        """
        df = pd.read_parquet(filepath).iloc[indices]
        features, labels = df.drop('label', axis=1), df['label']
        np_features = Client.scaler.fit_transform(features.to_numpy('float32'))
        np_labels = labels.to_numpy(bool).astype('float32')

        tensor_features = torch.from_numpy(np_features)
        tensor_labels = torch.from_numpy(np_labels)
        return tensor_features.to(self.device), tensor_labels.to(self.device)

    def train(
            self,
            train_set: tuple[Tensor, Tensor],
            server_state_dict: OrderedDict[str, Tensor],
            profile_on: bool
    ) -> tuple[float, int]:
        """
        Trains local model modified by the toml configuration file 
        (such as continual learning and differential privacy), 
        updates model, and calculates the average loss.
        """
        profile_on = False
        if profile_on:
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            with profile(activities=activities, profile_memory=True) as prof:
                result = self._train(train_set, server_state_dict)
            TRACE_PATH.write_text(
                prof.key_averages().table(sort_by='cpu_time_total')
            )
        else:
            result = self._train(train_set, server_state_dict)
        return result

    def _train(
            self,
            train_set: tuple[Tensor, Tensor],
            server_state_dict: OrderedDict[str, Tensor]
    ) -> tuple[float, int]:
        loop_start = time()
        self.update_model(server_state_dict)
        server_parameters = [
            param.detach().clone()
            for param in self.model.parameters()
        ]
        self.model.train()
        if self.cl_enabled:
            train_set = self.cl.er.sample_replay_buffer(
                original_dataset=train_set,
                n_new_samples=len(train_set[1]),
                ratio_old_samples=self.er_mix_ratio
            )
        data_loader = DataLoader(
            dataset=TensorDataset(*train_set),
            batch_size=self.batch_size,
            shuffle=True
        )
        data_loader = cast(DataLoader[tuple[Tensor, Tensor]], data_loader)
        # PrivacyEngine objects wrap around the original objects
        # N forward passes + N backward passes per batch (N = batch size)
        packages = self._create_model_packages(data_loader)
        training_model, optimizer, data_loader = packages
        scheduler = self.model.get_scheduler(
            optimizer=optimizer,
            cosine_epochs=len(data_loader) * self.epochs
        )

        total_loss = self._zero_tensor()
        total_samples = 0
        for _ in range(self.epochs):
            loss, n_samples = self._train_iteration(
                model=training_model,
                optimizer=optimizer,
                scheduler=scheduler,
                data_loader=data_loader,
                server_parameters=server_parameters
            )
            total_loss += loss
            total_samples += n_samples

        # Unwrap if dp enabled
        if isinstance(training_model, GradSampleModule):
            self.model = cast(MLP, training_model.to_standard_module())

        if self.cl_enabled:
            self.cl.er.add_data(train_set, self.er_sample_rate)
            self.cl.ewc.update_fisher_information(
                model=self.model,
                train_set=train_set,
                criterion=Client.criterion,
                batch_size=self.batch_size
            )
            self.cl.ewc.update_prev_parameters(self.model)
        average_loss = total_loss / max(1, total_samples)
        print(f"Training Time: {time() - loop_start:.3f} sec")
        return average_loss.item(), total_samples // self.epochs

    def _train_iteration(
            self,
            model: MLP | GradSampleModule,
            optimizer: Adam | DPOptimizer,
            scheduler: CosineAnnealingLR,
            data_loader: DataLoader[tuple[Tensor, Tensor]],
            server_parameters: Iterable[Tensor]
    ) -> tuple[Tensor, int]:
        n_samples = 0
        iteration_loss = self._zero_tensor()
        for batch in data_loader:
            batch_features, batch_labels = cast(tuple[Tensor, Tensor], batch)
            # Poisson sampling could produce empty batch
            if not batch_labels.nelement():
                continue

            outputs: Tensor = model(batch_features) # Forward pass
            outputs = outputs.squeeze(1)
            loss: Tensor = Client.criterion(outputs, batch_labels)

            # As the model changes from its gradients,
            # calculate penalties if new params stray too far
            if self.cl_enabled and not self.cl.ewc.is_empty():
                batch_fisher_penalty = self.cl.ewc.calculate_penalty(
                    model=model,
                    ewc_lambda=self.ewc_lambda
                )
                loss += batch_fisher_penalty

            # If FedAvg, ignore, else perform FedProx calculation
            if self.mu:
                proximal_term = self._calculate_proximal_term(
                    model=model,
                    server_parameters=server_parameters
                )
                loss += proximal_term

            optimizer.zero_grad() # Prevents gradient accumulation
            loss.backward() # Calculate where to step using mini-batch SGD
            optimizer.step() # Step forward down gradient
            scheduler.step() # Adjusts learning rate

            batch_size = len(batch_labels)
            n_samples += batch_size
            iteration_loss += loss.detach() * batch_size
        return iteration_loss, n_samples

    def _calculate_proximal_term(
            self,
            model: MLP | GradSampleModule,
            server_parameters: Iterable[Tensor]
    ) -> Tensor:
        # (mu / 2) * sum(||local - global||^2)
        proximal_term = self._zero_tensor()
        parameter_pairs = zip(model.parameters(), server_parameters)
        for client_parameter, server_parameter in parameter_pairs:
            difference = client_parameter - server_parameter
            # sum of squared vectors produces the l2 norm
            proximal_term += torch.sum(difference ** 2)
        return self.mu * 0.5 * proximal_term

    @torch.no_grad()
    def evaluate(
        self,
        test_set: tuple[Tensor, Tensor],
        server_parameters: OrderedDict[str, Tensor]
    ) -> tuple[dict[str, float], int]:
        """
        Loads server model and evaluates server aggregated model
        against the test set.
        """
        self.update_model(server_parameters)
        self.model.eval()
        data_loader = cast(
            DataLoader[tuple[Tensor, Tensor]],
            DataLoader(TensorDataset(*test_set), self.batch_size)
        )
        packages = self._create_model_packages(data_loader)
        evaluation_model, _, data_loader = packages

        data: dict[str, list[Tensor]] = {
            'labels': [],
            'predictions': [],
            'probabilities': []
        }
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
            batch_loss: Tensor = Client.criterion(outputs, batch_labels)
            total_loss += batch_loss * len(batch_labels)
            total_samples += len(batch_labels)

            data['labels'].append(batch_labels)
            data['predictions'].append(batch_predictions)
            data['probabilities'].append(batch_probabilities)

        if isinstance(evaluation_model, GradSampleModule):
            self.model = cast(MLP, evaluation_model.to_standard_module())

        metrics: dict[str, float] = {
            'accuracy': (n_correct / total_samples).item(),
            'loss': (total_loss / total_samples).item(),
        }
        fed_metrics = self._create_metrics(
            labels=torch.cat(data['labels']).cpu(),
            predictions=torch.cat(data['predictions']).cpu(),
            probabilities=torch.cat(data['probabilities']).cpu()
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
        fed_cl_ids_metrics = {
            'auroc': FedMetrics.auroc(labels, probabilities),
            'auprc': FedMetrics.auprc(labels, probabilities),
            'macro-f1': FedMetrics.macro_f1(labels, predictions),
            'recall@fpr=1%': FedMetrics.recall_at_fpr(labels, probabilities, 0.01),
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
        if not self.dp_enabled:
            return self.model, self.model.get_optimizer(), data_loader
        # Needs to be in train for make_private()
        if is_evaluating := not self.model.training:
            self.model.train()
        dp_model, dp_optimizer, *_, dp_data_loader = self.dp.make_private(
            module=self.model,
            optimizer=self.model.get_optimizer(),
            data_loader=data_loader,
            noise_multiplier=self.noise,
            max_grad_norm=self.clipping # Clipping value
        )
        if is_evaluating:
            dp_model.eval()
        return dp_model, dp_optimizer, dp_data_loader

    def _zero_tensor(self) -> Tensor:
        """Using tensor variables to reduce Tensor.item() sync overhead."""
        return torch.tensor(0.0, dtype=torch.float32, device=self.device)


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
    filepath = Path(cast(str, msg.content['config']['filepath']))
    row_indices = cast(list[int], msg.content['config']['flows'])
    train_set = client.data_from_indices(filepath, row_indices)
    average_loss, n_samples = client.train(train_set, server_parameters, profile_on)
    metrics = {
        'train_loss': average_loss,
        'epsilon': client.dp.get_epsilon(client.delta),
        'num-examples': n_samples
    }
    # state_dict() detaches from grad, but still on cuda
    client_parameters = cast(OrderedDict, client.model.state_dict())
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
    filepath = Path(cast(str, msg.content['config']['filepath']))
    row_indices: list[int] = cast(list[int], msg.content['config']['flows'])
    test_set = client.data_from_indices(filepath, row_indices)

    eval_metrics, n_samples = cast(
        tuple[dict[str, MetricRecordValues], int],
        client.evaluate(test_set, server_parameters)
    )
    eval_metrics['num-examples'] = n_samples
    content = RecordDict({'metrics': MetricRecord(eval_metrics)})
    return Message(content, reply_to=msg)
