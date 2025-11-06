from torch import Tensor
import torch
import numpy as np
import os

class ReplayBuffer:
    '''
    Maps the replay buffer to disk.
    The runtime folder should be cleared every run
    '''
    def __init__(self, identifier: int | str, n_features: int, path: str = "fed_cl_ids/runtime") -> None:
        self.path = path
        self.node_id = identifier
        self.n_features = n_features
    
        self.features_path = os.path.join(path, f"{identifier}_features.dat")
        self.labels_path = os.path.join(path, f"{identifier}_labels.dat")
        self._length_path: str = f"{self.path}/{identifier}_length.txt"

        # Memory mapped np arrays
        self._features: np.memmap
        self._labels: np.memmap

        # Initializes txt file
        if not os.path.exists(self._length_path):
            with open(self._length_path, 'w') as file:
                file.write('0')

    def append(self, features: Tensor, labels: Tensor) -> None:
        np_features = features.detach().numpy().astype('float32')
        np_labels = labels.detach().numpy().astype('float32')

        current_length = len(self)
        new_length = current_length + len(labels)

        mode = 'r+' if current_length else 'w+'

        # Initialize new size
        new_features = np.memmap(
            self.features_path, 
            dtype='float32', 
            mode=mode,
            shape=(new_length, self.n_features)
        )
        new_labels = np.memmap(
            self.labels_path, 
            dtype='float32', 
            mode=mode,
            shape=(new_length,)
        )
        
        # Copy old tensor data
        new_features[current_length:] = np_features
        new_labels[current_length:] = np_labels
        
        # Copy new tensor data
        if current_length > 0:
            new_features[:current_length] = self._features
            new_labels[:current_length] = self._labels

        # Write changes in the array to respective .dat file.
        new_features.flush()
        new_labels.flush()

        with open(self._length_path, 'w') as file:
            file.write(str(new_length))

        self._features = new_features
        self._labels = new_labels

    def _open_mmap(self) -> None:
        length = len(self)
        features_shape = (length, self.n_features)
        labels_shape = (length,)

        self._features = np.memmap(self.features_path, 'float32', 'r+', shape=features_shape)
        self._labels = np.memmap(self.labels_path, 'float32', 'r+', shape=labels_shape)

    def sample(self, n_samples: int) -> tuple[Tensor, Tensor]:
        length = len(self)
        if not length:
            raise ValueError("EMPTY REPLAY BUFFER")
        if n_samples > length:
            raise ValueError("SAMPLE SIZE TOO BIG")

        self._open_mmap()
        sample_indices = np.random.choice(length, n_samples, replace=False)
        features = torch.from_numpy(self._features[sample_indices].copy())
        labels = torch.from_numpy(self._labels[sample_indices].copy())
        return features, labels

    def __len__(self) -> int:
        with open(self._length_path) as file:
            return int(file.read())
        
    def __bool__(self) -> bool:
        return bool(len(self))
