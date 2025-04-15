import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_extractor import FeatureExtractor  # noqa: E402
from src.ahr_estimator import AHREstimator # noqa: E402
from utils.config import config

dfdc_subsets_v2_path = '/scratch/zceerba/DATASETS/DFDC_subsets'


# Set device
if torch.cuda.is_available():
    DEVICE_NUM = config.GPU
    torch.cuda.set_device(DEVICE_NUM)
    DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
else:
    DEVICE = torch.device("cpu")

chosen_subsets = ["subset_01", "subset_02", "subset_03", "subset_04"]

def main():
    # Define classes
    feature_extractor = FeatureExtractor()
    ahr_estimator = AHREstimator("/scratch/zceerba/projectSERN/audio_hr_v2/checkpoints/best_lstm_model.pth", device=DEVICE)

    for subset_folder in os.listdir(dfdc_subsets_v2_path):
        if 'subset_' in subset_folder and subset_folder in chosen_subsets:
            subset_folder_path = os.path.join(dfdc_subsets_v2_path, subset_folder)

            for split_folder in os.listdir(subset_folder_path):
                if split_folder in ['train', 'val', 'test']:

                    split_folder_path = os.path.join(subset_folder_path, split_folder)

                    # empty list for the split folder in the specific subset
                    subset_split_list = []

                    # Iterate through each video file in the split folder
                    for video_file in tqdm(os.listdir(split_folder_path), desc=f"Processing {subset_folder} - {split_folder}", leave=True):
                        if video_file.endswith(".mp4"):
                            video_path = os.path.join(split_folder_path, video_file)

                            # Estimate AHR
                            try:
                                ahr = ahr_estimator.estimate_hr(video_path)
                                error = None
                            except Exception as e:
                                ahr = None
                                error = e

                            # Create a dictionary for the video file and its AHR
                            video_dict = {
                                'video_file': video_file,
                                'ahr': ahr.tolist() if ahr is not None else None,
                                'error': error
                            }

                            # Append the dictionary to the list
                            subset_split_list.append(video_dict)

                    # Store in a json file
                    json_filepath = os.path.join(split_folder_path, f"{subset_folder}_{split_folder}_ahr.json")

                    with open(json_filepath, 'w') as json_file:
                        json.dump(subset_split_list, json_file, indent=4)

    print("DONE")


def main_npy():
    """
    Version 2 of the main function to save the AHR as numpy files.
    """
    # Define classes
    feature_extractor = FeatureExtractor()
    ahr_estimator = AHREstimator("/scratch/zceerba/projectSERN/audio_hr_v2/checkpoints/best_lstm_model.pth", device=DEVICE)

    for subset_folder in os.listdir(dfdc_subsets_v2_path):
        if 'subset_' in subset_folder and subset_folder in chosen_subsets:
            subset_folder_path = os.path.join(dfdc_subsets_v2_path, subset_folder)

            for split_folder in os.listdir(subset_folder_path):
                if split_folder in ['train', 'val', 'test']:

                    split_folder_path = os.path.join(subset_folder_path, split_folder)

                    if not os.path.exists(os.path.join(split_folder_path, "a_hr_seqs")):
                        os.makedirs(os.path.join(split_folder_path, "a_hr_seqs"))

                    # Iterate through each video file in the split folder
                    for video_file in tqdm(os.listdir(split_folder_path), desc=f"Processing {subset_folder} - {split_folder}", leave=True):
                        if video_file.endswith(".mp4"):
                            video_path = os.path.join(split_folder_path, video_file)
                            video_name  = video_file.split(".")[0]

                            # Estimate AHR
                            try:
                                ahr = ahr_estimator.estimate_hr(video_path)
                                error = None
                            except Exception as e:
                                ahr = None
                                error = e
                            # Save the AHR as a numpy file
                            np.save(os.path.join(split_folder_path, "a_hr_seqs", f"{video_name}.npy"), ahr)    

    print("DONE")

if __name__ == "__main__":
    main_npy()

