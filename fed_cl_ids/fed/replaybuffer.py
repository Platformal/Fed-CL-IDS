from torch import Tensor
import numpy as np
import torch
import os

class ReplayBuffer:
    """Maps the replay buffer to disk.
    The runtime folder should be cleared every run"""
    def __init__(
            self, identifier: int | str, np_dtype: str = 'float32',
            path: str = os.path.join("fed_cl_ids", "runtime")) -> None:
        self.dtype = np_dtype
        self._length: int = 0
        
        self.features_path = os.path.join(path, f"{identifier}_features.dat")
        self.labels_path = os.path.join(path, f"{identifier}_labels.dat")

        # Memory mapped, works as any ndarray
        self._features: np.memmap
        self._labels: np.memmap

    def sample(self, n_samples: int) -> tuple[Tensor, Tensor]:
        if not self._length:
            raise ValueError("Empty replay buffer.")
        if n_samples > self._length:
            raise ValueError("Sample size larger than replay buffer size.")

        sample_indices = np.random.choice(self._length, n_samples, replace=False)
        features = torch.from_numpy(self._features[sample_indices].copy())
        labels = torch.from_numpy(self._labels[sample_indices].copy())
        return features, labels

    def append(self, features: Tensor, labels: Tensor) -> None:
        new_length = self._length + len(labels)
        writing_mode = 'r+' if self._length else 'w+'

        # Kept features and labels separate since you don't have to
        # convert labels to 2D -> 1D and vice versa when sampling
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
        new_features[self._length:] = features.detach()
        new_labels[self._length:] = labels.detach()
        
        # Write changes in the array to respective .dat file.
        new_features.flush()
        new_labels.flush()

        self._features = new_features
        self._labels = new_labels
        self._length = new_length

    def __len__(self) -> int:
        return self._length
        
    def __bool__(self) -> bool:
        return bool(len(self))
