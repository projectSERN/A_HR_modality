import os
import sys
import torch

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
        self.model = LSTMHiddenSummation(in_dim=13, hidden_size=64, num_layers=4, dropout=0.4, out_dim=1)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()

    def estimate_hr(self, path: str):
        # Extract MFCCs from video or audio path
        mfccs = self.extractor.feature_extraction(path)

        # Transform into tensor
        mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0).to(self.device) # add batch dimension

        # Estimate HR
        with torch.no_grad():
            hr = self.model(mfccs_tensor).squeeze(0).detach().cpu().numpy()

        return hr
