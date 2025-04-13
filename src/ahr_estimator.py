import os
import sys
import torch
import numpy as np

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_extractor import FeatureExtractor
from src.models import LSTMHiddenSummation

class AHREstimator:
    def __init__(self, model_path: str, device):
        self.model_path = model_path
        self.device = device
        self.extractor = FeatureExtractor()
        self.__load_model()


    def __load_model(self):
        self.model = LSTMHiddenSummation(in_dim=13, hidden_size=32, num_layers=5, dropout=0.4, out_dim=1)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(self.device)
        self.model.eval()


    def estimate_hr(self, path: str, clip_length: int = None):
        # Extract MFCCs from video or audio path
        mfccs = self.extractor.feature_extraction(path)

        # Transform into tensor
        mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0).to(self.device) # add batch dimension

        # If clip_length is not provided, use the entire sequence
        if clip_length: 
            mfccs_tensors = torch.split(mfccs_tensor, clip_length, dim=1)
            mfccs_tensors = [tensor for tensor in mfccs_tensors if tensor.shape[1] == clip_length]

        # Estimate HR
        hr_values = []
        with torch.no_grad():
            if clip_length:
                for tensor in mfccs_tensors:
                    hr = self.model(tensor).squeeze(0).detach().cpu().numpy()
                    hr_values.append(hr)
                return hr_values
            else:
                hr = self.model(mfccs_tensor).squeeze(0).detach().cpu().numpy()
                return hr
