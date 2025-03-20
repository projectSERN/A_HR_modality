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


class LSTMHiddenSummation(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, out_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc= nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        # Take the last hidden state
        hidden = hidden[-1].unsqueeze(1) # (batch, 1, hidden_size)
        hidden = hidden.expand(-1, output.shape[1], -1) # (batch, seq_len, hidden_size)

        cell = cell[-1].unsqueeze(1) # (batch, 1, hidden_size)
        cell = cell.expand(-1, output.shape[1], -1) # (batch, seq_len, hidden_size) 

        fused = output + hidden + cell
        # Pass the hidden state summed with the output to the fully connected layer
        output = self.fc(fused)
        return output
