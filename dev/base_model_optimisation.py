import os
import sys
from numpy.typing import ArrayLike
from typing import List
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# /home/zceerba/.conda/envs/projectSERN/bin/python

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models import LSTM, LSTMHiddenSummation, CNN_LSTM # noqa: E402
from src.data_preprocessor import DataPreprocessor # noqa: E402
from utils.early_stopping import EarlyStopping # noqa: E402
from utils.config import config

# Set device
if torch.cuda.is_available():
    DEVICE_NUM = config.GPU
    torch.cuda.set_device(DEVICE_NUM)
    DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
else:
    DEVICE = torch.device("cpu")
RANDOM_SEED = 7

SERN_V3_PATH = "/home/zceerba/projectSERN/DATASETS/PROJECTSERN_DATASET/v3/base_dataset_v3.npz"
SERN_V4_PATH = "/home/zceerba/projectSERN/DATASETS/PROJECTSERN_DATASET/v4/base_dataset_v4.npz"
KCON_PATH = "/home/zceerba/projectSERN/DATASETS/K-EMOCON/base_dataset_kcon.npz"

torch.manual_seed(RANDOM_SEED)

processor = DataPreprocessor()
sern_data_v3 = processor.load_dataset(SERN_V3_PATH)
sern_data_v4 = processor.load_dataset(SERN_V4_PATH)
sern_data = sern_data_v3 + sern_data_v4

kcon_data = processor.load_dataset(KCON_PATH)

datasets = [kcon_data]

def create_clipped_dataset(data: List[ArrayLike], clip_length: int):
    sets = []
    for dataset in data:
        clipped = processor.create_clipped_dataset(dataset, clip_length=clip_length)
        scaled = processor.scale_feature_inputs(clipped)
        tensor = torch.tensor(scaled, device=DEVICE, dtype=torch.float32)
        X, y = tensor[:, :, :-1], tensor[:, :, -1].unsqueeze(-1)
        sets.append([(X[i], y[i]) for i in range(len(X))])

    return sets


def get_dataloaders(datasets: List[ArrayLike], batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for the train, test, and validation datasets.

    Arg(s):
    - datasets (List[ArrayLike]): The datasets to be converted into DataLoaders
    - batch_size (int): The batch size for the DataLoaders

    Returns:
    - List[DataLoader]: The DataLoaders for the train, test, and validation datasets
    """
    dataloaders = []
    for dataset in datasets:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        dataloaders.append(loader)
    return dataloaders


def objective(trial, datasets: List[ArrayLike]):
    train_losses = []
    val_losses = []
    val_rmses = []

    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32, 64])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])
    clip_length = 10
    num_layers = trial.suggest_int("num_layers", 2, 5)
    cnn_channels = trial.suggest_categorical("cnn_channels", [8, 16, 32, 64])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4])

    # Create dataset
    datasets = create_clipped_dataset(datasets, clip_length=clip_length)
    train_dataset, val_dataset = train_test_split(datasets[0], test_size=0.3, random_state=RANDOM_SEED)
    val_dataset, test_dataset = train_test_split(val_dataset, test_size=1/3, random_state=RANDOM_SEED)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Instantiate LSTM model
    lstm = LSTMHiddenSummation(in_dim=13, hidden_size=hidden_size, out_dim=1, num_layers=num_layers, dropout=dropout)
    lstm = lstm.to(DEVICE)
    loss_func = nn.HuberLoss(delta=0.5)
    optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=3)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, mode="min")

    epochs = 100
    # Training loop
    for epoch in range(epochs):
        lstm.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimiser.zero_grad()
            hr_train_pred = lstm(x_batch)
            loss = loss_func(hr_train_pred, y_batch)
            train_loss += loss.item()
            loss.backward()
            optimiser.step()

        # Take the average loss from training each batch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluation
        lstm.eval()
        val_loss = 0.0
        val_rmse = 0.0
        with torch.no_grad(): # Disable gradient tracking and computation
            for x_batch, y_batch in val_loader:
                optimiser.zero_grad()
                hr_val_pred = lstm(x_batch)
                loss = loss_func(hr_val_pred, y_batch)
                rmse = root_mean_squared_error(y_batch.cpu().reshape(-1, 1), hr_val_pred.cpu().reshape(-1, 1))
                val_loss += loss.item()
                val_rmse += rmse

        # Take the average loss from validating each batch
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_rmse /= len(val_loader)
        val_rmses.append(val_rmse)

        # Reduce learning rate if validation loss plateaus
        scheduler.step(val_loss)

        if epoch % 5 == 0:
            print(f"Epoch: {epoch} | Train Loss: {train_losses[-1]: .2f} | Validation Loss: {val_losses[-1]: .2f}")

        early_stopping(val_rmse)
        if early_stopping.early_stop:
            print(f"Early stopping at Epoch {epoch}")
            break

    return val_rmses[-1]

study = optuna.create_study(direction="minimize")
study.optimize(lambda trial: objective(trial, datasets), n_trials=100, show_progress_bar=True)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
print(study.best_params)