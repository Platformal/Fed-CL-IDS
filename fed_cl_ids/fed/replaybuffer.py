import numpy as np
from torch import Tensor
import torch
import os

class ReplayBuffer:
    '''Maps the replay buffer to disk.
    The runtime folder should be cleared every run'''
    def __init__(
            self, identifier: int | str, 
            n_features: int,
            path: str = os.path.join("fed_cl_ids", "runtime")) -> None:
        self._n_features = n_features
        self._length: int = 0
        
        self._features_path = os.path.join(path, f"{identifier}_features.dat")
        self._labels_path = os.path.join(path, f"{identifier}_labels.dat")
        self._length_path = os.path.join(path, f"{identifier}_length.txt")

        # Memory mapped, works as any ndarray
        self._features: np.memmap
        self._labels: np.memmap

    def sample(self, n_samples: int) -> tuple[Tensor, Tensor]:
        if not self._length:
            raise ValueError("Empty replay buffer.")
        if n_samples > self._length:
            raise ValueError("Sample size larger than replay buffer size.")

        self._open_mmap()
        sample_indices = np.random.choice(self._length, n_samples, replace=False)
        features = torch.from_numpy(self._features[sample_indices].copy())
        labels = torch.from_numpy(self._labels[sample_indices].copy())
        return features, labels
    
    def _open_mmap(self) -> None:
        self._features = np.memmap(
            filename=self._features_path,
            dtype='float32',
            mode='r+',
            shape=(self._length, self._n_features)
        )
        self._labels = np.memmap(
            self._labels_path,
            dtype='float32',
            mode='r+',
            shape=(self._length,)
        )

    def append(self, features: Tensor, labels: Tensor) -> None:
        np_features = features.detach().numpy().astype('float32')
        np_labels = labels.detach().numpy().astype('float32')

        new_length = self._length + len(labels)
        mode = 'r+' if self._length else 'w+'

        # Don't want it to create a new size every call
        # Use something similar to arraylist; double capacity
        # Could also combine them into one .dat file
        new_features = np.memmap(
            self._features_path,
            dtype='float32',
            mode=mode,
            shape=(new_length, self._n_features)
        )
        new_labels = np.memmap(
            self._labels_path, 
            dtype='float32',
            mode=mode,
            shape=(new_length,)
        )
        
        # Copy old tensor data
        if self._length:
            new_features[:self._length] = self._features
            new_labels[:self._length] = self._labels
            
        # Copy new tensor data
        new_features[self._length:] = np_features
        new_labels[self._length:] = np_labels
        

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
