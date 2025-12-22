"""Holds multilayer perceptron model."""
from typing import Iterable

from opacus.optimizers.optimizer import DPOptimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.adam import Adam
from torch import nn, Tensor

class MLP(nn.Module):
    """Main model used by server and clients for federated learning."""
    def __init__(
            self,
            n_features: int,
            hidden_widths: Iterable[int],
            dropout: float,
            weight_decay: float,
            lr_max: float,
            lr_min: float
        ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_widths = hidden_widths
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.network = self._create_network()

    def _create_network(self) -> nn.Sequential:
        layers = []
        prev_dimension = self.n_features # Number of inputs
        for width in self.hidden_widths:
            layers.extend((
                nn.Linear(prev_dimension, width), # Hidden layer
                nn.LayerNorm(width),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ))
            prev_dimension = width
        layers.append(nn.Linear(prev_dimension, 1)) # Final neuron output
        return nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        """
        Pass inputs through the model for training/prediction.
        
        :param features: Features with dimension 1 == self.n_features.
        :type features: Tensor
        :return: Tensor of shape (-1, 1) where -1 is number of inputs
        (e.g. batch size 128 outputs tensor of (128, 1)).
        :rtype: Tensor
        """
        return self.network(features)

    def get_optimizer(self) -> Adam:
        """
        Creates new Adam optimizer (due to changing parameters).
        
        :return: Optimizer with learning rate and weight decay tuned from
        configuration file.
        :rtype: Adam
        """
        optimizer = Adam(
            params=self.parameters(),
            lr=self.lr_max,
            weight_decay=self.weight_decay
        )
        return optimizer

    def get_scheduler(self, optimizer: Adam | DPOptimizer, cosine_epochs: int) -> CosineAnnealingLR:
        """
        Create new cosine scheduler.
        
        :param optimizer: Adam optimizer from self.get_optimizer()
        :type optimizer: Adam
        :param cosine_epochs: Determines how scheduler modifies learning rate.
        Suggested num_batches * num_epochs from `here <https://discuss.pytorch.org/t/cosineannealinglr-step-size-t-max/104687/5>`__.
        :type cosine_epochs: int
        :return: Scheduler.
        :rtype: CosineAnnealingLR
        """
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=self.lr_min,
            T_max=cosine_epochs
        )
        return scheduler
