"""Fed-CL-IDS: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


# class MLP(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self):
#         super(MLP, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

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

    def forward(self, x) -> None:
        self.network(x)
    
    def get_optimizer(self, n_iterations: int) -> tuple:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr_max,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=self.lr_min,
            T_max=n_iterations
        )
        return optimizer, scheduler

fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(model, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    model.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(model, testloader, device):
    """Validate the model on the test set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
