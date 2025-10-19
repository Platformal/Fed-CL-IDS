import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import pandas as pd

# Should return test and train sets
def split_uavids(df: pd.DataFrame, flows: list[int], n_batches: int):
    filtered = df.loc[flows]
    features, labels = filtered.drop('label', axis=1), filtered['label']
    features_tensor = torch.from_numpy(features.to_numpy())
    labels_tensor = torch.from_numpy(labels.to_numpy())
    dataset = TensorDataset(features_tensor, labels_tensor)

    train_ratio, test_ratio = 0.8, 0.2
    train_set, test_set = random_split(dataset, (train_ratio, test_ratio),
                                       torch.Generator().manual_seed(0))
    train = DataLoader(train_set, batch_size=n_batches, shuffle=True)
    test = DataLoader(test_set, batch_size=n_batches, shuffle=True)
    return train, test

class MLP(nn.Module):
    # Values should be initialized by the server and passed onto clients
    def __init__(
            self,
            n_features: int =  0, 
            n_classes: int = 0, 
            hidden_widths: list[int] = [], 
            dropout: float = 0.0, 
            weight_decay: float = 0.0,
            lr_max: float = 0.0, 
            lr_min: float = 0.0) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_widths = hidden_widths
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.network = self._create_network()

    def _create_network(self) -> nn.Sequential:
        layers = []
        prev_dimension = self.n_features # Input dimensions
        for width in self.hidden_widths:
            layer = nn.Linear(prev_dimension, width)
            layers.extend((
                layer,
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ))
            prev_dimension = width
        output_layer = nn.Linear(prev_dimension, self.n_classes) # Final output
        layers.append(output_layer)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> None:
        # self.network(x)
        return self.network(x)
    
    def get_optimizer(self, n_iterations: int) -> tuple[Adam, CosineAnnealingLR]:
        optimizer = Adam(
            self.parameters(),
            lr=self.lr_max,
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=self.lr_min,
            T_max=n_iterations
        )
        return optimizer, scheduler

def client_train(model: MLP, train_data: DataLoader, n_epochs: int, device: torch.device) -> float:
    model.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer, scheduler = model.get_optimizer(len(train_data) * 20)
    model.train()
    running_loss = 0.0
    for _ in range(n_epochs):
        for batch in train_data:
            features, labels = batch
            # Move inputs/labels to the correct device and dtypes
            features = features.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.long)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(train_data)
    return avg_trainloss

def client_test(model: MLP, test_data: DataLoader, device: torch.device):
    model.to(device)
    # Use CrossEntropyLoss for a model that outputs logits for each class
    loss_evaluator = nn.CrossEntropyLoss().to(device)
    correct, total_loss = 0, 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in test_data:
            features, labels = batch
            features = features.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.long)
            outputs = model(features)
            total_loss += loss_evaluator(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    # Compute averages
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / len(test_data)
    return avg_loss, accuracy
