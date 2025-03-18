import os
import sys
from typing import Dict, List
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.custom_datasets import SequenceDataset # noqa: E402
from src.audio_manipulator import AudioManipulator # noqa: E402
from src.feature_extractor import FeatureExtractor # noqa: E402

class DataPreprocessor:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.manipulator = AudioManipulator()


    def aggregate_dataset_paths(self, base_dir: str) -> Dict[str, str]:
        data = {}
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            #print(folder_path)

            # Look through each scenario (exercise, no_exercise) folders from the base directory filepath
            if os.path.isdir(folder_path):
                for child_folder in os.listdir(folder_path):
                    child_folder_path = os.path.join(folder_path, child_folder)

                    # Look inside individual sample folders
                    if os.path.isdir(child_folder_path):
                        video_file = [file for file in os.listdir(child_folder_path) if file.lower().endswith(".mov")][0]
                        csv_file = [file for file in os.listdir(child_folder_path) if file.lower().endswith(".csv")][0]

                        # Relative path to each file
                        video_path = os.path.join(child_folder_path, video_file)
                        csv_path = os.path.join(child_folder_path, csv_file)
                        
                        data[video_path] = csv_path
        return data


    def create_base_dataset(self, datapaths: Dict[str, str], save: bool = False, filepath: str = None, deltas: bool = False) -> ArrayLike:        
        # Empty list to store feature arrays
        features = []

        for video_path in datapaths.keys():
            # Extract audio from video and reduce noise
            audio_signal = self.manipulator.extract_audio(video_path)
            cleaned_audio_signal = self.manipulator.reduce_noise(audio_signal, 22050)

            # Calculate MFCCs from cleaned audio signal
            mfccs = self.extractor.find_mfccs(cleaned_audio_signal, hop_length=22050)

            # Remove extra frame to match length of HR signal
            mfccs = mfccs[:, :-1]

            # Transpose MFCC array - (13 coefficients, N) -> (N, 13)
            mfccs = mfccs.T

            if deltas:
                # Calculate deltas and delta-deltas
                delta_mfccs = self.extractor.find_deltas(cleaned_audio_signal, order=1, hop_length=22050)
                delta2_mfccs = self.extractor.find_deltas(cleaned_audio_signal, order=2, hop_length=22050)
                delta_mfccs = delta_mfccs[:, :-1].T
                delta2_mfccs = delta2_mfccs[:, :-1].T

                # Concatenate MFCCs, deltas, and delta-deltas
                mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=1)

            # Get HR signal
            csv_path = datapaths[video_path]
            hr_signal = pd.read_csv(csv_path)["HR"].to_numpy().reshape(-1, 1)

            # Concatenate MFCCs and HR signal and append to list of features
            features.append(np.concatenate((mfccs, hr_signal), axis=1))

        # Save dataset to file
        if save:
            np.savez(filepath, *features)

        return features


    def create_clipped_dataset(self, dataset: List[np.array], clip_length: int) -> ArrayLike:
        samples = []
        for feature_array in dataset:
            feature_clips = self.__split_data(feature_array, clip_length)
            sample_features = np.stack(feature_clips, axis=0)
            samples.append(sample_features)

        return np.concatenate(samples, axis=0)


    def get_sequence_lengths(self, dataset: List[ArrayLike]) -> ArrayLike:
        sequence_lengths = []
        for feature_array in dataset:
            sequence_length = feature_array.shape[0]
            sequence_lengths.append(sequence_length)
        return sequence_lengths


    def create_sequence_dataset(self, dataset: List[ArrayLike]) -> SequenceDataset:
        sequence_lengths = self.get_sequence_lengths(dataset)
        sequence_dataset = SequenceDataset(dataset, sequence_lengths)
        return sequence_dataset


    def __split_data(self, data: np.array, clip_length):
        # Input data must be 2D and splits along the first axis (0)
        if data.ndim != 2:
            raise ValueError(f"Expected amount of dimensions is 2, but got {data.ndim}")
        # clip length will be set to 30 seconds for now
        num_clips = len(data) // clip_length
        clips = [data[i * clip_length: (i + 1) * clip_length, :] for i in range(num_clips)]
        return clips
    

    def load_dataset(self, filepath: str) -> ArrayLike:
        loaded_dataset = np.load(filepath)
        loaded_dataset = [loaded_dataset[key] for key in loaded_dataset.files]
        return loaded_dataset


    def create_window_dataset(self, data: List[ArrayLike], window_size: int, offset: int) -> List[ArrayLike]:
        windows = []
        for sample in data:
            time_steps = sample.shape[0]
            for start in range(0, time_steps - window_size + 1, offset):
                window = sample[start: start + window_size]
                windows.append(window)
        return windows

    def create_sliding_window_dataset(self, data: List[ArrayLike], window_size: int, offset: int = 1) -> List[ArrayLike]:
        """
        Create a dataset with sliding windows

        Arg(s):
        - data (List[ArrayLike]): List of 2D arrays, where each array has shape (time steps, features + target)
        - window_size (int): Size of the window
        - offset (int): Number of steps to move the window by
        """
        X, y = [], []
        for array in data:
            num_samples = (len(array) - window_size) // offset + 1

            for i in range(0, num_samples * offset, offset):
                # Ensure the window fits within the current array
                if i + window_size <= len(array):
                    X.append(array[i: i + window_size, :-1]) # Features (MFCCs)
                    y.append(np.mean(array[i: i + window_size, -1])) # Target (HR)

        return np.array(X), np.array(y)


    def scale_data(self, data, type: str = "minmax", scale_target: bool = False):
        # Set up scalers
        feature_scaler = StandardScaler()
        target_scaler = MinMaxScaler()
        # if type == "minmax":
        #     feature_scaler = MinMaxScaler()
        #     target_scaler = MinMaxScaler()
        # elif type == "standard":
        #     feature_scaler = StandardScaler()
        #     target_scaler = StandardScaler()

        # Scale the data
        for i, sample in enumerate(data):
            hr_signal = sample[:, -1].reshape(-1, 1)
            mfcc_values = sample[:, :-1]
            scaled_mffcs  = feature_scaler.fit_transform(mfcc_values)
            if scale_target:
                hr_signal = target_scaler.fit_transform(hr_signal)
            scaled_combined = np.concatenate((scaled_mffcs, hr_signal), axis=1)
            data[i] = scaled_combined
        return data, target_scaler


    def scale_sequences(self, mfcc_frames: List[torch.tensor], hr_signals: List[torch.tensor]) -> List[torch.tensor]:
        all_mfccs = torch.cat(mfcc_frames)
        mfccs_min = all_mfccs.min()
        mfccs_max = all_mfccs.max()

        all_hr = torch.cat(hr_signals)
        hr_min = all_hr.min()
        hr_max = all_hr.max()

        scaled_mfccs = [(seq - mfccs_min) /  (mfccs_max - mfccs_min) for seq in mfcc_frames]
        scaled_hr = [(seq - hr_min) /  (hr_max - hr_min) for seq in hr_signals]

        combined_sequences = []

        # Concatenate the MFCCs and HR together
        for i in range(len(mfcc_frames)):
            new_tensor = torch.cat((scaled_mfccs[i], scaled_hr[i]), dim=1)
            combined_sequences.append(new_tensor)
        
        return combined_sequences, hr_min, hr_max, mfccs_min, mfccs_max
    

    def reverse_hr_scaling(self, hr_signal: ArrayLike, min_value: float, max_value: float) -> ArrayLike:
        return hr_signal * (max_value - min_value) + min_value