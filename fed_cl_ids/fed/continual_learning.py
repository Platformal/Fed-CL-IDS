"""Module for continual learning using experience replay and
elastic weight consolidation."""
from pathlib import Path

from opacus.grad_sample.grad_sample_module import GradSampleModule
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss
from torch import Tensor
import torch
import numpy as np

from fed_cl_ids.models.mlp import MLP

class ReplayBuffer:
    """Maps the replay buffer to disk as a numpy memory-mapped array 
    (essentially a ndarray).
    
    The runtime folder should be cleared every run to prevent file name 
    conflicts."""
    def __init__(
            self,
            identifier: int | str,
            path: Path,
            np_dtype: str = 'float32') -> None:
        self.dtype = np_dtype
        self._length: int = 0

        # Two separate files since you don't have to
        # convert labels to 2D -> 1D and vice versa when sampling
        self.features_path = path / f'{identifier}_features.dat'
        self.labels_path = path / f'{identifier}_labels.dat'

        # Memory mapped, works as any ndarray
        self._features: np.memmap
        self._labels: np.memmap

    def sample(self, n_samples: int) -> tuple[Tensor, Tensor]:
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

        sample_indices = np.random.choice(self._length, n_samples, replace=False)
        features = torch.from_numpy(self._features[sample_indices].copy())
        labels = torch.from_numpy(self._labels[sample_indices].copy())
        return features, labels

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

        new_features = np.memmap(
            self.features_path,
            dtype=self.dtype,
            mode=writing_mode,
            shape=(new_length, features.shape[1])
        )
        new_labels = np.memmap(
            self.labels_path,
            dtype=self.dtype,
            mode=writing_mode,
            shape=(new_length,)
        )

        # Copy old tensor data
        if self._length:
            new_features[:self._length] = self._features
            new_labels[:self._length] = self._labels

        # Copy new tensor data
        new_features[self._length:] = features.cpu().detach()
        new_labels[self._length:] = labels.cpu().detach()

        # Write changes in the array to respective .dat file.
        new_features.flush()
        new_labels.flush()

        self._features = new_features
        self._labels = new_labels
        self._length = new_length

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
            lambda_penalty: int | float) -> Tensor:
        # Penalty = (λ/2) * Σ F_i(θ_i - θ*_i)²
        # Since minibatch, we multiply by (λ/2) for each sum item
        ewc_loss = torch.tensor(0.0)
        for name, parameter in model.named_parameters():
            # Private model prepends _module. to each param name
            name = name.removeprefix("_module.")
            if name not in self._prev_parameters:
                continue # Never triggered
            f_i = self._fisher_diagonal[name]
            nested_term = parameter - self._prev_parameters[name]
            nested_term = nested_term.pow(2)
            penalty = f_i * nested_term
            ewc_loss += penalty.sum()
        return (lambda_penalty / 2) * ewc_loss

    def update_fisher_information(
            self,
            model: MLP,
            train_set: tuple[Tensor, Tensor],
            criterion: BCEWithLogitsLoss,
            batch_size: int) -> None:
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
            name: torch.zeros_like(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        data = TensorDataset(*train_set)
        batches = DataLoader(data, batch_size=batch_size)
        for batch_features, batch_labels in batches:
            batch_features: Tensor = batch_features
            batch_labels: Tensor = batch_labels

            model.zero_grad()
            outputs: Tensor = model(batch_features)
            outputs = outputs.squeeze(1)
            loss: Tensor = criterion(outputs, batch_labels)
            loss.backward()

            # Sum of square gradients
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        new_fisher[name] += parameter.grad.detach().pow(2)

        for name in new_fisher:
            new_fisher[name] /= len(batches)
        self._fisher_diagonal = new_fisher

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