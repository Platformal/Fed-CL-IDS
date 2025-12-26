"""Module for continual learning using experience replay
and elastic weight consolidation."""
from dataclasses import dataclass
from pathlib import Path
import math

from opacus.grad_sample.grad_sample_module import GradSampleModule
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss
from torch import Tensor
import torch

from models.mlp import MLP

class ContinualLearning:
    """Container for both experience replay and elastic weight consolidation"""
    def __init__(
            self,
            er_filepath_identifier: int | str,
            er_runtime_directory: Path,
            device: torch.device
    ) -> None:
        self.ewc = ElasticWeightConsolidation(device)
        self.er = ExperienceReplay(
            filepath_identifier=er_filepath_identifier,
            runtime_directory=er_runtime_directory,
            device=device
        )


class ElasticWeightConsolidation:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._prev_parameters: dict[str, Tensor] = {}
        self._fisher_diagonal: dict[str, Tensor] = {}

    def calculate_penalty(
            self,
            model: MLP | GradSampleModule,
            ewc_lambda: int | float
    ) -> Tensor:
        # Penalty = (λ/2) * Σ F_i(θ_i - θ*_i)²
        sum_loss = torch.tensor(0.0, device=self.device)
        for name, parameter in model.named_parameters():
            # Private model prepends _module. to each param name
            name = name.removeprefix('_module.')
            if name not in self._prev_parameters:
                print("Parameter not in prev_parameters")
                continue # Never triggered (since ewc.is_empty())
            f_i = self._fisher_diagonal[name]
            theta_star = self._prev_parameters[name]
            penalty_i = f_i * (parameter - theta_star).pow(2)
            sum_loss += penalty_i.sum()
        return ewc_lambda * 0.5 * sum_loss

    def update_fisher_information(
            self,
            model: MLP,
            train_set: tuple[Tensor, Tensor],
            criterion: BCEWithLogitsLoss,
            batch_size: int
    ) -> None:
        """
        Calculates the fisher diagonal from the current model
        and training data.
        
        :param train_set: Same features and labels used in training the model.
        :type train_set: tuple[Tensor, Tensor]
        :return: Calculated fisher diagonal from modified/trained parameters.
        :rtype: dict[str, Tensor]
        """
        model.eval()
        if not isinstance(model, MLP):
            raise TypeError("MLP model should be passed")

        new_fisher: dict[str, Tensor] = {
            name: torch.zeros_like(parameter, device=self.device)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        data_loader = DataLoader(
            dataset=TensorDataset(*train_set),
            batch_size=batch_size
        )

        for batch_features, batch_labels in data_loader:
            batch_features: Tensor
            batch_labels: Tensor

            model.zero_grad()
            outputs: Tensor = model(batch_features)
            outputs = outputs.squeeze(1)
            loss: Tensor = criterion(outputs, batch_labels)
            loss.backward()

            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    square_gradients = parameter.grad.detach().pow(2)
                    new_fisher[name] += square_gradients * len(batch_labels)

        total_samples = len(train_set[1])
        for name in new_fisher:
            new_fisher[name] /= total_samples

        if not self._fisher_diagonal:
            self._fisher_diagonal = new_fisher
            return
        for name, parameter in new_fisher.items():
            self._fisher_diagonal[name] += parameter

    def update_prev_parameters(self, model: MLP) -> None:
        if not isinstance(model, MLP):
            raise TypeError("MLP model should be passed")
        self._prev_parameters = {
            name: parameter.clone().detach()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def is_empty(self) -> bool:
        return not bool(self._prev_parameters)


class ExperienceReplay:
    def __init__(
            self,
            filepath_identifier: int | str,
            runtime_directory: Path,
            device: torch.device
    ) -> None:
        self.device = device
        self._buffer = ReplayBuffer(
            identifier=filepath_identifier,
            path=runtime_directory
        )

    def sample_replay_buffer(
            self,
            original_dataset: tuple[Tensor, Tensor],
            n_new_samples: int,
            ratio_old_samples: float,
    ) -> tuple[Tensor, Tensor]:
        if not ratio_old_samples or not n_new_samples:
            return original_dataset
        if ratio_old_samples >= 1.00:
            raise ValueError("Experience replay cannot be >= 100%")
        # Could also do: (new_samples * 0.2) / 0.8 = 20% of total (mix = 0.2)
        true_ratio = (1 - ratio_old_samples) / ratio_old_samples
        ideal_size = int(n_new_samples / true_ratio)
        if not (actual_size := min(len(self._buffer), ideal_size)):
            return original_dataset
        # Ratio will be off until replay buffer size >= replay sample size
        new_dataset = self._buffer.sample(actual_size)
        return self._concat_datasets(original_dataset, new_dataset)

    def _concat_datasets(
            self,
            tensor_data: tuple[Tensor, Tensor],
            sampled_tensors: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        new_tensors = tuple(map(lambda x: x.to(self.device), sampled_tensors))
        features, labels = map(torch.cat, zip(tensor_data, new_tensors))
        return features, labels

    def add_data(
            self,
            original_dataset: tuple[Tensor, Tensor],
            sample_rate: float,
    ) -> None:
        # Prefer exact n_samples than random per sampling
        features, labels = original_dataset
        n_dataset_samples = len(labels)
        if n_samples := int(n_dataset_samples * sample_rate):
            selected = torch.randint(
                0, n_dataset_samples,
                size=(n_samples,),
                device=self.device
            )
        else:
            uniform_values = torch.rand((n_dataset_samples,), device=self.device)
            selected = uniform_values <= sample_rate
        selected_features = features[selected].cpu().detach()
        selected_labels = labels[selected].cpu().detach()
        self._buffer.append(selected_features, selected_labels)

@dataclass
class MemoryMappedData:
    def __init__(self, filepath: Path) -> None:
        self.path = filepath
        self.n_inputs: int
        self.memmap: Tensor

class ReplayBuffer:
    """Maps the replay buffer to disk as a numpy memory-mapped array 
    (essentially a ndarray). All operations using replay buffer is respective
    to numpy and its arrays.
    
    The runtime folder should be cleared every run to prevent file name 
    conflicts."""
    INITIAL_CAPACITY = 256
    def __init__(
            self,
            identifier: int | str,
            path: Path,
            dtype: torch.dtype = torch.float32
    ) -> None:
        self.dtype = dtype
        self._features = MemoryMappedData(path / f'{identifier}_features.bin')
        self._labels = MemoryMappedData(path / f'{identifier}_labels.bin')
        self._length = 0
        self._capacity = 0

    def append(self, features: Tensor, labels: Tensor) -> None:
        if not labels.nelement():
            return

        if not hasattr(self._labels, 'memmap'):
            self._capacity = ReplayBuffer.INITIAL_CAPACITY
            self._features.n_inputs = features.shape[1]
            self._labels.n_inputs = 1
            self._features.memmap = self._new_memmap(self._features, self._capacity)
            self._labels.memmap = self._new_memmap(self._labels, self._capacity)

        new_length = self._length + len(labels)
        if new_length > self._capacity:
            self._capacity = 1 << math.ceil(math.log2(new_length))
            self._increase_memmap(self._features, self._capacity)
            self._increase_memmap(self._labels, self._capacity)

        self._features.memmap[self._length:new_length].copy_(features)
        self._labels.memmap[self._length:new_length].copy_(labels)
        self._length = new_length

    def _increase_memmap(self, data_obj: MemoryMappedData, new_capacity: int) -> None:
        new_memmap = self._new_memmap(data_obj, new_capacity)
        new_memmap[:self._length].copy_(data_obj.memmap[:self._length])
        data_obj.memmap = new_memmap

    def _new_memmap(self, data_obj: MemoryMappedData, length: int) -> Tensor:
        new_memmap = torch.from_file(
            filename=str(data_obj.path),
            shared=True,
            # Takes number of elements, not rows
            size=length * data_obj.n_inputs,
            dtype=self.dtype
        )
        if data_obj.n_inputs > 1:
            shape = (-1, data_obj.n_inputs)
            new_memmap = new_memmap.reshape(shape)
        return new_memmap

    def sample(self, n_samples: int) -> tuple[Tensor, Tensor]:
        if not self._length:
            raise ValueError("Empty replay buffer.")
        if n_samples > self._length:
            raise ValueError("Sample size larger than replay buffer size.")

        indices = torch.randperm(self._length)[:n_samples]
        features = self._features.memmap[indices]
        labels = self._labels.memmap[indices]
        return features, labels

    def __len__(self) -> int:
        return self._length

    def __bool__(self) -> bool:
        return bool(self._length)
