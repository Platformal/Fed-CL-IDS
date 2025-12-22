"""Module for continual learning using experience replay and
elastic weight consolidation."""
from typing import Literal, Optional
from pathlib import Path

from opacus.grad_sample.grad_sample_module import GradSampleModule
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss
from torch import Tensor
import torch
import numpy as np

from fed_cl_ids.models.mlp import MLP

class ContinualLearning:
    """Container for both experience replay and elastic weight consolidation"""
    def __init__(
            self,
            er_filepath_identifier: int | str,
            er_runtime_directory: Path
    ) -> None:
        self.ewc = ElasticWeightConsolidation()
        self.er = ExperienceReplay(
            filepath_identifier=er_filepath_identifier,
            runtime_directory=er_runtime_directory
        )

class ExperienceReplay:
    def __init__(
            self,
            filepath_identifier: int | str,
            runtime_directory: Path
    ) -> None:
        self._buffer = ReplayBuffer(
            identifier=filepath_identifier,
            path=runtime_directory
        )

    def sample_replay_buffer(
            self,
            original_dataset: tuple[Tensor, Tensor],
            n_new_samples: int,
            ratio_old_samples: float,
            device: torch.device
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
        np_dataset = self._buffer.sample(actual_size)
        return self._concat_samples(original_dataset, np_dataset, device)

    def _concat_samples(
            self,
            original_dataset: tuple[Tensor, Tensor],
            np_dataset: tuple[np.ndarray, np.ndarray],
            device: torch.device
    ) -> tuple[Tensor, Tensor]:
        np_features, np_labels = np_dataset
        new_features = torch.from_numpy(np_features).to(device)
        new_labels = torch.from_numpy(np_labels).to(device)
        original_features, original_labels = original_dataset
        concat_features = torch.cat((original_features, new_features))
        concat_labels = torch.cat((original_labels, new_labels))
        return concat_features, concat_labels

    def add_data(
            self,
            original_dataset: tuple[Tensor, Tensor],
            sample_rate: float,
            device: torch.device
    ) -> None:
        features, labels = original_dataset
        # Prefer exact n_samples than random per sampling
        if n_samples := int(len(labels) * sample_rate):
            selected = torch.randint(
                low=0,
                high=len(labels),
                size=(n_samples,),
                device=device
            )
        else:
            uniform_values = torch.rand(
                size=(len(labels),),
                device=device
            )
            selected = uniform_values <= sample_rate
        selected_features = features[selected].cpu().detach()
        selected_labels = labels[selected].cpu().detach()
        self._buffer.append(selected_features, selected_labels)

class ReplayBuffer:
    """Maps the replay buffer to disk as a numpy memory-mapped array 
    (essentially a ndarray). All operations using replay buffer is respective
    to numpy and its arrays.
    
    The runtime folder should be cleared every run to prevent file name 
    conflicts."""
    def __init__(
            self,
            identifier: int | str,
            path: Path,
            np_dtype: str = 'float32'
    ) -> None:
        self.dtype = np_dtype
        self._length: int = 0

        # Two separate files since you don't have to
        # convert labels to 2D -> 1D and vice versa when sampling
        self.features_path = path / f'{identifier}_features.dat'
        self.labels_path = path / f'{identifier}_labels.dat'

        # Memory mapped, works as any ndarray
        self._features: np.memmap
        self._labels: np.memmap

    def sample(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly samples from buffer for tensors.
        
        :param n_samples: How many samples to retrieve. Will raise an error if
        the replay buffer is empty or n_samples is bigger than buffer.
        :type n_samples: int
        :return: Tensor of features and labels.
        :rtype: tuple[Tensor, Tensor]
        """
        if not self._length:
            raise ValueError("Empty replay buffer.")
        if n_samples > self._length:
            raise ValueError("Sample size larger than replay buffer size.")

        indices = np.random.choice(self._length, n_samples, replace=False)
        np_features = self._features[indices].copy()
        np_labels = self._labels[indices].copy()
        return np_features, np_labels

    def append(self, features: Tensor, labels: Tensor) -> None:
        """
        Add features and labels into memory mapped files. Will early return if
        size of rows is 0.
        
        :param features: Make sure columns of features is consistent.
        :type features: Tensor
        :param labels: Make sure number of labels matches rows of features.
        :type labels: Tensor
        """
        if not labels.nelement():
            return
        new_length = self._length + len(labels)
        writing_mode = 'r+' if self._length else 'w+'
        self._features = self._copy_to_new_memmap(
            path=self.features_path,
            writing_mode=writing_mode,
            new_shape=(new_length, features.shape[1]),
            old_data=getattr(self, '_features', None),
            new_data=features
        )
        self._labels = self._copy_to_new_memmap(
            path=self.labels_path,
            writing_mode=writing_mode,
            new_shape=(new_length,),
            old_data=getattr(self, '_labels', None),
            new_data=labels
        )
        self._length = new_length

    def _copy_to_new_memmap(
            self,
            path: Path,
            writing_mode: Literal['r+', 'w+'],
            new_shape: tuple,
            old_data: Optional[np.memmap],
            new_data: np.ndarray | Tensor
    ) -> np.memmap:
        new_memmap = np.memmap(
            filename=path,
            dtype=self.dtype,
            mode=writing_mode,
            shape=new_shape
        )
        if old_data is not None:
            new_memmap[:self._length] = old_data
        new_memmap[self._length:] = new_data
        new_memmap.flush() # Updates .dat file with new memmap
        return new_memmap

    def __len__(self) -> int:
        return self._length

    def __bool__(self) -> bool:
        return bool(self._length)


class ElasticWeightConsolidation:
    def __init__(self) -> None:
        self._prev_parameters: dict[str, Tensor] = {}
        self._fisher_diagonal: dict[str, Tensor] = {}

    def calculate_penalty(
            self,
            model: MLP | GradSampleModule,
            lambda_penalty: int | float,
            device: torch.device
    ) -> Tensor:
        # Penalty = (λ/2) * Σ F_i(θ_i - θ*_i)²
        sum_loss = torch.tensor(0.0, device=device)
        for name, parameter in model.named_parameters():
            # Private model prepends _module. to each param name
            name = name.removeprefix("_module.")
            if name not in self._prev_parameters:
                print("Parameter not in prev_parameters")
                continue # Never triggered
            f_i = self._fisher_diagonal[name]
            theta_star = self._prev_parameters[name]
            penalty_i = f_i * (parameter - theta_star).pow(2)
            sum_loss += penalty_i.sum()
        return lambda_penalty * 0.5 * sum_loss

    def update_fisher_information(
            self,
            model: MLP,
            train_set: tuple[Tensor, Tensor],
            criterion: BCEWithLogitsLoss,
            batch_size: int,
            device: torch.device) -> None:
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
            name: torch.zeros_like(parameter, device=device)
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

            # Sum of square gradients
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    square_gradients = parameter.grad.detach().pow(2)
                    new_fisher[name] += square_gradients * len(batch_labels)

        total_samples = len(train_set[1]) # Should equal samples in dataloader
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
