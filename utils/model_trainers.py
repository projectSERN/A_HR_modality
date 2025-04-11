"""
This module contains the LSTMTrainer classes which is used to train
the various LSTM models.
"""
import os
import sys
from numpy.typing import ArrayLike
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score
from scipy.stats import pearsonr
from dtaidistance import dtw
from frechetdist import frdist
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.early_stopping import EarlyStopping # noqa: E402


class ModelTrainer:
    def __init__(self, train_loader: ArrayLike,
                 test_loader: ArrayLike,
                 val_loader: ArrayLike,
                 optimiser: torch.optim,
                 scheduler: torch.optim.lr_scheduler,
                 loss_function: torch.nn,
                 model: nn.Module,
                 epochs: int):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.lf = loss_function
        self.model = model
        self.epochs = epochs
        self.train_losses = []
        self.val_losses = []


    def plot_loss_curves(self, epoch_resolution: int, path: str) -> None:
        sampled_epochs = list(range(0, len(self.train_losses), epoch_resolution))
        sampled_train_losses = self.train_losses[::epoch_resolution]
        sampled_val_losses = self.val_losses[::epoch_resolution]

        plt.plot(sampled_epochs, sampled_train_losses, label="Training loss")
        plt.plot(sampled_epochs, sampled_val_losses, label="Validation loss")
        plt.grid()
        plt.title("Training and loss curves during training")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        if path:
            plt.savefig(path)
        plt.show()


    def display_test_results(self, test_results: dict[str, float]) -> None:
        console = Console()
        table = Table(title="Test Metrics")

        table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
        table.add_column("Value", justify="center", style="magenta")

        for metric, value in test_results.items():
            table.add_row(metric, f"±{value: .2f}")

        console.print(table)


class LSTMTrainer(ModelTrainer):
    def __init__(self, train_loader: ArrayLike,
                 test_loader: ArrayLike,
                 val_loader: ArrayLike,
                 optimiser: torch.optim,
                 scheduler: torch.optim.lr_scheduler,
                 loss_function: torch.nn,
                 model: nn.Module,
                 epochs: int):
        super().__init__(train_loader, test_loader, val_loader, optimiser, scheduler, loss_function, model, epochs)
        
    def train(self, patience: int, delta: int = 0.0) -> nn.Module:

        # Define early stopping
        early_stopping = EarlyStopping(patience=patience, delta=delta)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for x_batch, y_batch in self.train_loader:
                self.optimiser.zero_grad()
                hr_train_pred = self.model(x_batch)
                loss = self.lf(hr_train_pred, y_batch)
                train_loss += loss.item()
                loss.backward()
                self.optimiser.step()

            # Take the average loss from training each batch
            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)

            # Evaluation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad(): # Disable gradient tracking and computation
                for x_batch, y_batch in self.val_loader:
                    self.optimiser.zero_grad()
                    hr_val_pred = self.model(x_batch)
                    loss = self.lf(hr_val_pred, y_batch)
                    val_loss += loss.item()

            # Take the average loss from validating each batch
            val_loss /= len(self.val_loader)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if epoch % 5 == 0:
                print(f"Epoch: {epoch} | Train Loss: {self.train_losses[-1]: .2f} | Validation Loss: {self.val_losses[-1]: .2f}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at Epoch {epoch}")
                break

        return self.model


    def evaluate(self, target_scaler: None, display: bool = False) -> tuple[float, float, float, float, float]:
        # Test 
        self.model.eval()
        with torch.no_grad():
            rmses = []
            maes = []
            pearsonr_coeffs = []
            dtw_distances = []
            frdists =  []
            mapes = []
            for x_batch, y_batch in self.test_loader:
                hr_test_pred = self.model(x_batch)

                if display:
                    # Plot the HR predictions against the ground truth
                    # plot the first sample in each batch
                    if target_scaler is None:
                        pred = hr_test_pred[0].cpu().detach().numpy()
                        gt = y_batch[0].cpu().detach().numpy()
                    elif target_scaler:
                        pred = target_scaler.inverse_transform(hr_test_pred[0].cpu().detach().numpy())
                        gt = target_scaler.inverse_transform(y_batch[0].cpu().detach().numpy())
                    plt.plot(pred, label="Predicted HR")
                    plt.plot(gt, label="Ground Truth HR")
                    plt.legend()
                    plt.title("Predicted HR vs Ground Truth HR")
                    plt.xlabel("Time (secs)")
                    plt.ylabel("HR (bpm)")
                    plt.show()

                if target_scaler is None:
                    hr_test_pred = hr_test_pred.reshape(-1, 1).cpu().detach().numpy()
                    hr_ground_truth = y_batch.reshape(-1, 1).cpu().detach().numpy()
                else:
                    hr_test_pred = target_scaler.inverse_transform(hr_test_pred.reshape(-1, 1).cpu().detach().numpy())
                    hr_ground_truth = target_scaler.inverse_transform(y_batch.reshape(-1, 1).cpu().detach().numpy())

                # Calculate metrics
                rmse = root_mean_squared_error(hr_ground_truth.reshape(-1,1), hr_test_pred.reshape(-1, 1))
                mae = mean_absolute_error(hr_ground_truth.reshape(-1,1), hr_test_pred.reshape(-1, 1))
                pearsonr_coeff, _ = pearsonr(hr_ground_truth.reshape(-1), hr_test_pred.reshape(-1))
                dtw_distance = dtw.distance(hr_ground_truth.reshape(-1), hr_test_pred.reshape(-1))
                frdist_distance = frdist(hr_ground_truth, hr_test_pred)
                mape = (mae / np.mean(hr_ground_truth.reshape(-1))) * 100

                rmses.append(rmse)
                maes.append(mae)
                mapes.append(mape)
                pearsonr_coeffs.append(pearsonr_coeff)
                dtw_distances.append(dtw_distance)
                frdists.append(frdist_distance)

        test_rmse = np.mean(np.array(rmses))
        test_mae = np.mean(np.array(maes))
        test_mape = np.mean(np.array(mapes))
        test_pearsonr = np.mean(np.array(pearsonr_coeffs))
        test_dtw = np.mean(np.array(dtw_distances))
        test_frdist = np.mean(np.array(frdists))

        results = {
            "RMSE": test_rmse,
            "MAE": test_mae,
            "MAPE": test_mape,
            "PCC": test_pearsonr,
            "DTW": test_dtw,
            "FRDist": test_frdist
        }

        self.display_test_results(results)


class EncoderTrainer(ModelTrainer):
    def __init__(self, train_loader, test_loader, val_loader, optimiser, scheduler, loss_function, model, epochs):
        super().__init__(train_loader, test_loader, val_loader, optimiser, scheduler, loss_function, model, epochs)
        self.val_accuracies = []

    def pre_train(self, patience: int, delta: int = 0.0) -> nn.Module:
        # Define early stopping
        early_stopping = EarlyStopping(patience=patience, delta=delta)

        # Training loop
        for epoch in tqdm(range(self.epochs), desc="Pre-training encoder", leave=True):

            self.model.train()
            train_loss = 0.0
            for x_batch, y_batch in self.train_loader:
                self.optimiser.zero_grad()
                preds, _= self.model(x_batch)
                loss = self.lf(preds.squeeze(), y_batch)
                train_loss += loss.item()
                loss.backward()
                self.optimiser.step()

            # Take the average loss from training each batch
            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)

            # Evaluation
            self.model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():
                for x_batch, y_batch in self.val_loader:
                    self.optimiser.zero_grad()
                    preds, _ = self.model(x_batch)
                    loss = self.lf(preds.squeeze(), y_batch)
                    val_loss += loss.item()
                    # Calculate accuracy
                    predictions = (preds > 0.5).float()
                    accuracy = accuracy_score(y_batch.cpu(), predictions.cpu())
                    val_accuracy += accuracy

            # Take the average loss from validating each batch
            val_loss /= len(self.val_loader)
            self.scheduler.step(val_loss)
            self.val_losses.append(val_loss)
            val_accuracy /= len(self.val_loader)
            self.val_accuracies.append(val_accuracy)

            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break

            if epoch % 5 == 0:
                print(f"Epoch: {epoch} | Train Loss: {self.train_losses[-1]: .3f} | Val Loss: {self.val_losses[-1]: .3f} | Val Accuracy: {self.val_accuracies[-1]: .3f}")
        print("\n")


    def evaluate_pre_training(self):
        self.model.eval()
        with torch.no_grad():
            test_accuracy = 0.0
            test_roc_auc = 0.0
            for x_batch, y_batch in self.test_loader:
                self.optimiser.zero_grad()
                preds, _ = self.model(x_batch)
                predictions = (preds > 0.5).float()
                roc_auc = roc_auc_score(y_batch.cpu(), predictions.cpu())
                accuracy = accuracy_score(y_batch.cpu(), predictions.cpu())
                test_accuracy += accuracy
                test_roc_auc += roc_auc

        test_accuracy /= len(self.test_loader)
        test_roc_auc /= len(self.test_loader)
        print(f"Test accuracy: {test_accuracy * 100: .3f} %")
        print(f"Test ROC-AUC: {test_roc_auc * 100: .3f} %")
        print("\n")
