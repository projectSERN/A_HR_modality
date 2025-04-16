import os
import sys
from functools import partial
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models import AHR_ConvEncoder, AHR_LSTMEncoder # noqa: E402
from utils.early_stopping import EarlyStopping # noqa: E402
from utils.custom_datasets import collate_encoder_fn # noqa: E402
from utils.config import config

# Set device
if torch.cuda.is_available():
    DEVICE_NUM = config.GPU
    torch.cuda.set_device(DEVICE_NUM)
    DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
else:
    DEVICE = torch.device("cpu")

RANDOM_SEED = 7
MODEL = config.MODEL

# Define path to dataset for encoder pre-training
ITW_PATH = "/scratch/zceerba/DATASETS/release_in_the_wild/full_dataset_v2.npz"

# Define random seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load data
data = np.load(ITW_PATH, allow_pickle=True)

# Get the features and targets
features = data["features"]
labels = data["labels"]
print("Features and targets loaded")

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = np.array([scaler.fit_transform(feature_array) for feature_array in features], dtype=np.float32)
print("Features scaled")


# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=RANDOM_SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/8, random_state=RANDOM_SEED)
print("Data split into training and validation sets")

# Create DataLoaders
train_set = list(zip(X_train, y_train))
val_set = list(zip(X_val, y_val))
test_set = list(zip(X_test, y_test))

loss_func = nn.BCELoss()

def objective(trial):
    """
    Objective function for Optuna optimization.
    """
    # Training hyperparameters to optimize
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3, 1e-2])
    lambda2 = trial.suggest_float('lambda2', 1e-6, 0.9, log=True)

    # LSTM encoder hyperparameters
    if MODEL == "lstm":
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])
        dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])
        num_layers = trial.suggest_categorical("lstm_layers", [2, 3, 4, 5, 6])

    # Convolutional encoder hyperparameters
    elif MODEL == "conv":
        kernel_size = trial.suggest_categorical("kernel_size", [2, 3, 4, 5])
        padding = trial.suggest_categorical("padding", [1, 2, 3])

    epochs = 100

    partial_collate_fn = partial(collate_encoder_fn, device=DEVICE)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=partial_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=partial_collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=partial_collate_fn)

    # Initialize model
    if MODEL == "lstm":
        model = AHR_LSTMEncoder(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, num_features=1, num_classes=1).to(DEVICE)
    elif MODEL == "conv":
        model = AHR_ConvEncoder(kernel_size=kernel_size, padding=padding, num_features=1, num_classes=1).to(DEVICE)
    model = model.to(DEVICE)

    # Define loss function and optimizer
    optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda2)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=3)

    # Define early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.0, mode="max")

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_aurocs = []

    # Training loop
    for epoch in tqdm(range(epochs), desc="Pre-training encoder", leave=True):

        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimiser.zero_grad()
            preds, _ = model(x_batch)
            loss = loss_func(preds.squeeze(1), y_batch)
            train_loss += loss.item()
            loss.backward()
            optimiser.step()

        # Take the average loss from training each batch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_roc_auc = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                optimiser.zero_grad()
                preds, _ = model(x_batch)
                loss = loss_func(preds.squeeze(1), y_batch)
                val_loss += loss.item()

                # Calculate accuracy
                predictions = (preds > 0.5).float()
                accuracy = accuracy_score(y_batch.cpu(), predictions.cpu())

                # Check if both classes are present before calculating ROC AUC
                if len(torch.unique(y_batch)) > 1:
                    roc_auc = roc_auc_score(y_batch.cpu(), preds.cpu())
                else:
                    roc_auc = 0.5  # Assign a default value if only one class is present

                val_accuracy += accuracy
                val_roc_auc += roc_auc

        # Take the average loss from validating each batch
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_roc_auc /= len(val_loader)
        scheduler.step(val_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_aurocs.append(val_roc_auc)

        # Early stopping
        early_stopping(val_accuracy)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch: {epoch}")
            break

        if epoch % 5 == 0:
            print(f"Epoch: {epoch} | Train Loss: {train_losses[-1]: .3f} | Val Loss: {val_losses[-1]: .3f} | Val Accuracy: {val_accuracies[-1]: .3f}")

    print("\n")

    return val_accuracies[-1]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200, show_progress_bar=True)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
print(study.best_params)

