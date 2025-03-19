import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_preprocessor import DataPreprocessor # noqa: E402
from src.models import LSTM # noqa: E402
from utils.model_trainers import LSTMTrainer # noqa: E402

# Define constants
RANDOM_SEED = 12
EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 2
INPUT_SIZE = 13
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
NUM_LAYERS = 4
DROPOUT = 0
CLIP_LENGTH = 5

# Set device
if torch.cuda.is_available():
    DEVICE_NUM = 0
    torch.cuda.set_device(DEVICE_NUM)
    DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
else:
    DEVICE = torch.device("cpu")

SERN_V3_PATH = "../../DATASETS/PROJECTSERN_DATASET/v3/base_dataset_v3.npz"
SERN_V4_PATH = "../../DATASETS/PROJECTSERN_DATASET/v4/base_dataset_v4.npz"
KCON_PATH =  "../../DATASETS/K-EMOCON/base_dataset_kcon.npz"   


def main():
    # Define random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    processor = DataPreprocessor()

    sern_v3_data = processor.load_dataset(SERN_V3_PATH)
    sern_v4_data = processor.load_dataset(SERN_V4_PATH)
    sern_data = sern_v3_data + sern_v4_data

    # Create clipped dataset
    clipped_data = processor.create_clipped_dataset(sern_data, clip_length=CLIP_LENGTH)

    # Scaled data
    scaled_data = processor.scale_feature_inputs(clipped_data)

    # Put dataset into tensor
    tensor_data = torch.tensor(scaled_data, device=DEVICE, dtype=torch.float32)

    # Split data into features and targets
    features = tensor_data[:, :, :-1]
    targets = tensor_data[:, :, -1].unsqueeze(-1)

    # Split data into training, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=RANDOM_SEED)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=1/3, random_state=RANDOM_SEED)
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples, Val: {len(X_val)} samples")

    # Testing the train test val split function
    train_set = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    test_set = [(X_test[i], y_test[i]) for i in range(len(X_test))]
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    val_set = [(X_val[i], y_val[i]) for i in range(len(X_val))]
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    loss_func = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3)
    trainer = LSTMTrainer(train_loader, test_loader, val_loader, optimiser, scheduler, loss_func, model, EPOCHS)

    trainer.train(patience=5)
    trainer.evaluate(None, display=False)
    trainer.plot_loss_curves(epoch_resolution=1, path="")

if __name__ == "__main__":
    main()