"""
Functions to carry out cross-corpus analysis.
"""
import os
import sys
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_preprocessor import DataPreprocessor


def __create_cross_corpus_dataloaders(train_dataset: torch.Tensor, test_dataset: torch.Tensor, batch_size: int, random_seed: int):
    # Create the train and validation dataloaders
    train_dataset, _ = train_test_split(train_dataset, train_size=0.8, random_state=random_seed)
    train_dataset, val_dataset = train_test_split(train_dataset, train_size=0.8, random_state=random_seed)

    _, test_dataset = train_test_split(test_dataset, test_size=0.2, random_state=random_seed)
    
    # Split the datasets into features and targets
    X_train, y_train = train_dataset[:, :, :-1], train_dataset[:, :, -1].unsqueeze(-1)
    X_val, y_val = val_dataset[:, :, :-1], val_dataset[:, :, -1].unsqueeze(-1)
    X_test, y_test = test_dataset[:, :, :-1], test_dataset[:, :, -1].unsqueeze(-1)

    # Create the dataloaders
    train_set = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    val_set = [(X_val[i], y_val[i]) for i in range(len(X_val))]
    test_set = [(X_test[i], y_test[i]) for i in range(len(X_test))]

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, val_dataloader


def create_cross_corpus(datasets: dict[str, List[np.ndarray]],
                        clip_length: int,
                        batch_size: int,
                        device: str,
                        random_seed: int,
                        chosen_train_dataset: str,
                        reduce: bool = True
                        ) -> List[np.ndarray]:
    """
    Create a cross-corpus dataset by concatenating the datasets.
    
    Arg(s):
        - datasets (dict[str, List[np.ndarray]]): A dictionary containing the datasets to concatenate.
    
    Returns:
        - List[np.ndarray]: A list containing the concatenated datasets.
    """
    processor = DataPreprocessor()

    for key, dataset in datasets.items():
        datasets[key] = processor.create_clipped_dataset(dataset, clip_length=clip_length)
        datasets[key] = processor.scale_feature_inputs(datasets[key])
        datasets[key] = torch.tensor(datasets[key], device=device, dtype=torch.float32)

    if reduce:
        # Reduce the size of the KCON dataset (if arg passed)
        percentage = 0.6
        np.random.seed(random_seed)
        num_keep_samples = int(datasets["kcon"].shape[0] * percentage)
        random_indices = np.random.choice(datasets["kcon"].shape[0], num_keep_samples, replace=False)
        datasets["kcon"] = datasets["kcon"][random_indices] 
    
    if chosen_train_dataset == "kcon":
        # Use KCON as the training set
        train_dataset = datasets["kcon"]
        test_dataset = datasets["sern"]

    elif chosen_train_dataset == "sern":
        # Use SERN as the training set
        train_dataset = datasets["sern"]
        test_dataset = datasets["kcon"]

    train_loader, test_loader, val_loader = __create_cross_corpus_dataloaders(
        train_dataset=train_dataset, test_dataset=test_dataset, batch_size=batch_size, random_seed=random_seed)
    
    return train_loader, test_loader, val_loader
