import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_preprocessor import DataPreprocessor # noqa: E402
from src.models import LSTM, LSTMHiddenSummation, CNN_LSTM # noqa: E402
from utils.model_trainers import LSTMTrainer # noqa: E402
from utils.corpus_helpers import create_cross_corpus # noqa: E402
from utils.config import config

# Define constants
RANDOM_SEED = 7
EPOCHS = 50
LEARNING_RATE = config.LEARNING_RATE
BATCH_SIZE = config.BATCH_SIZE
INPUT_SIZE = 13
HIDDEN_SIZE = config.HIDDEN_SIZE
OUTPUT_SIZE = 1
NUM_LAYERS = config.NUM_LAYERS
DROPOUT = config.DROPOUT
CLIP_LENGTH = config.CLIP_LENGTH
CNN_CHANNELS = 16

# Set device
if torch.cuda.is_available():
    DEVICE_NUM = config.GPU
    torch.cuda.set_device(DEVICE_NUM)
    DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
else:
    DEVICE = torch.device("cpu")

SERN_V3_PATH = "/home/zceerba/projectSERN/DATASETS/PROJECTSERN_DATASET/v3/base_dataset_v3.npz"
SERN_V4_PATH = "/home/zceerba/projectSERN/DATASETS/PROJECTSERN_DATASET/v4/base_dataset_v4.npz"
KCON_PATH = "/home/zceerba/projectSERN/DATASETS/K-EMOCON/base_dataset_kcon.npz" 
SUBSET = config.SUBSET

def main():
    # Define random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    processor = DataPreprocessor()

    sern_v3_data = processor.load_dataset(SERN_V3_PATH)
    sern_v4_data = processor.load_dataset(SERN_V4_PATH)
    kcon_data = processor.load_dataset(KCON_PATH)
    sern_data = sern_v3_data + sern_v4_data

    datasets = {"sern": sern_data, "kcon": kcon_data}

    train_loader, test_loader, val_loader = create_cross_corpus(datasets=datasets,
                                                                clip_length=CLIP_LENGTH,
                                                                batch_size=BATCH_SIZE,
                                                                device=DEVICE,
                                                                random_seed=RANDOM_SEED,
                                                                chosen_train_dataset="sern",
                                                                reduce=SUBSET)
    
    # Instantiate model
    lstm = LSTMHiddenSummation(in_dim=INPUT_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    out_dim=OUTPUT_SIZE,
                    num_layers=NUM_LAYERS,
                    dropout=DROPOUT)
    lstm = lstm.to(DEVICE)

    # Define loss function and optimiser
    loss_func = nn.HuberLoss(delta=0.5)
    optimiser = optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=3)

    # Train model
    trainer = LSTMTrainer(model=lstm,
                          loss_function=loss_func,
                          optimiser=optimiser,
                          scheduler=scheduler,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          test_loader=test_loader,
                          epochs=EPOCHS,
                          device=DEVICE)
    
    trainer.train(patience=3)
    trainer.evaluate(None, display=False)
    trainer.plot_loss_curves(epoch_resolution=1, path="/home/zceerba/projectSERN/audio_hr_v2/cross_corpus_loss_curves.png")


if __name__ == "__main__":
    main()