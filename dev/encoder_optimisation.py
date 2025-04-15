import os
import sys
from functools import partial
from numpy.typing import ArrayLike
import numpy as np
from typing import List
import optuna
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models import AHR_ConvEncoder # noqa: E402
from utils.early_stopping import EarlyStopping # noqa: E402
from utils.model_trainers import EncoderTrainer # noqa: E402
from utils.custom_datasets import collate_encoder_fn # noqa: E402
from utils.config import config

# Set device
if torch.cuda.is_available():
    DEVICE_NUM = config.GPU
    torch.cuda.set_device(DEVICE_NUM)
    DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
else:
    DEVICE = torch.device("cpu")
RANDOM_SEED = 9

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

# Initialize model
model = AHR_ConvEncoder(num_features=1, num_classes=1)
model.to(DEVICE)

loss_func = nn.BCELoss()


def objective(trial):
    """
    Objective function for Optuna optimization.
    """
    # Define hyperparameters to optimize
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3, 1e-2])
    lambda2 = trial.suggest_float('lambda2', 1e-6, 0.9, log=True)
    epochs = 50

    partial_collate_fn = partial(collate_encoder_fn, device=DEVICE)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=partial_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=partial_collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=partial_collate_fn)

    # Initialize model
    model = AHR_ConvEncoder(num_features=1, num_classes=1).to(DEVICE)

    # Define loss function and optimizer
    optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda2)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=2)

    trainer = EncoderTrainer(train_loader, test_loader, val_loader, optimiser, scheduler, loss_func, model, epochs, DEVICE)
    trainer.pre_train(patience=5)

    return trainer.val_accuracies[-1]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
print(study.best_params)


