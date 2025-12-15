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
            new_data: Tensor) -> np.memmap:
        new_memmap = np.memmap(
            filename=path,
            dtype=self.dtype,
            mode=writing_mode,
            shape=new_shape
        )
        if old_data is not None:
            new_memmap[:self._length] = old_data
        new_memmap[self._length:] = new_data.cpu().detach()
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
            device: torch.device) -> Tensor:
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
