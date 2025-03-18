"""
This file contains the model definition for the LSTM model.
"""
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, out_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc= nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output
