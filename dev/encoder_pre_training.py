import os
import sys
from functools import partial
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

## /home/zceerba/.conda/envs/projectSERN/bin/python

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models import AHR_LSTMEncoder, AHR_ConvEncoder # noqa: E402
from src.model_trainers import EncoderTrainer # noqa: E402
from utils.custom_datasets import collate_encoder_fn # noqa: E402
from utils.config import config

# Define constants
RANDOM_SEED = 9
EPOCHS = config.EPOCHS
LEARNING_RATE = config.LEARNING_RATE
BATCH_SIZE = config.BATCH_SIZE
L2 = config.LAMBDA2
HIDDEN_SIZE = config.HIDDEN_SIZE
NUM_LAYERS = config.NUM_LAYERS
DROPOUT = config.DROPOUT
MODEL = config.MODEL
KERNEL_SIZE = config.KERNEL_SIZE
PADDING = config.PADDING

# Set device
if torch.cuda.is_available():
    DEVICE_NUM = config.GPU
    torch.cuda.set_device(DEVICE_NUM)
    DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
else:
    DEVICE = torch.device("cpu")

# Define path to dataset for encoder pre-training
ITW_PATH = "/scratch/zceerba/DATASETS/release_in_the_wild/full_dataset_v2.npz"

def main():
    # Define random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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

    partial_collate_fn = partial(collate_encoder_fn, device=DEVICE)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=partial_collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=partial_collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=partial_collate_fn)
    print("Dataloaders created")

    # Initialize model
    if MODEL == "lstm":
        model = AHR_LSTMEncoder(num_features=1, num_classes=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
    elif MODEL == "conv":
        model = AHR_ConvEncoder(num_features=1, num_classes=1, kernel_size=KERNEL_SIZE)
    model.to(DEVICE)

    # Initialize optimizer and loss function
    loss_func = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=2)

    trainer = EncoderTrainer(train_loader, test_loader, val_loader, optimiser, scheduler, loss_func, model, EPOCHS, DEVICE)
    print(f"Training configuration: BATCH SIZE = {BATCH_SIZE} | EPOCHS = {EPOCHS} | LEARNING RATE = {LEARNING_RATE} | L2 = {L2}" +
          f" | NUMBER OF LAYERS = {NUM_LAYERS} | HIDDEN SIZE = {HIDDEN_SIZE} | DROPOUT = {DROPOUT} | MODEL = {MODEL} | KERNEL SIZE = {KERNEL_SIZE} | PADDING = {PADDING}")
    trainer.pre_train(patience=5)
    trainer.plot_loss_curves(epoch_resolution=2, path="/scratch/zceerba/projectSERN/audio_hr_v2/encoder_loss_curves.png")
    trainer.evaluate_pre_training()


if __name__ == "__main__":
    main()