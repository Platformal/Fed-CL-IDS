from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.adam import Adam
from torch import nn, Tensor

class MLP(nn.Module):
    # Values should be initialized by the server and passed onto clients
    def __init__(
            self,
            n_features: int, 
            hidden_widths: list[int],
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

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)

    def get_optimizers(self, n_iterations: int) -> tuple[Adam, CosineAnnealingLR]:
        optimizer = Adam(
            params=self.parameters(),
            lr=self.lr_max,
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=self.lr_min,
            T_max=n_iterations
        )
        return optimizer, scheduler

    def _create_network(self) -> nn.Sequential:
        layers = []
        prev_dimension = self.n_features # Input dimensions
        for width in self.hidden_widths:
            layers.extend((
                nn.Linear(prev_dimension, width), # Main layer
                nn.LayerNorm(width),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ))
            prev_dimension = width
        output_layer = nn.Linear(prev_dimension, 1) # Final neuron output
        layers.append(output_layer)
        return nn.Sequential(*layers)