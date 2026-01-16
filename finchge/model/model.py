from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import re
warnings.filterwarnings("ignore")

class Model:
    """A wrapper class for training and managing PyTorch neural networks."""

    def __init__(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer | None = None,
        batch_size: int = 64,
        epochs: int = 50,
        device: torch.device | None = None,
    ) -> None:
        # Select device: use provided one, otherwise prefer GPU if available
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        self.net = net.to(self.device)
        print(self.net)

        # Define loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # Initialize optimizer if not provided
        self.optimizer = optimizer or optim.Adam(self.net.parameters(), lr=1e-5)

        self.num_epochs = epochs
        self.batch_size = batch_size


    def fit(self, train_dataset, val_dataset):
        print("Train the network after parsing the phenotype as a network.")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.net.train()  # Set to training mode at start of epoch
            for inputs, labels in train_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels.long())
                preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # average epoch loss
            epoch_loss = running_loss / len(train_loader)

    def predict(self, val_dataset):
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        self.net.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(self.device)
                val_labels = val_labels.to(self.device)
                val_outputs = self.net(val_inputs)

                # Calculate the validation loss
                val_loss += self.criterion(val_outputs, val_labels.long()).item()

                # Count correct predictions
                predicted_labels = torch.max(val_outputs, dim=1)[1]

                correct += (predicted_labels == val_labels).sum().item()

                # Move predictions and targets to CPU for metric calculation
                val_predictions.extend(predicted_labels.cpu().tolist())
                val_targets.extend(val_labels.cpu().tolist())
        return val_predictions, val_targets