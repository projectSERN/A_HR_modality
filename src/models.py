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


class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, cnn_channels, lstm_hidden, lstm_layers, output_dim, dropout, pool_size):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=cnn_channels, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=2)
        self.pool = nn.AdaptiveMaxPool2d(pool_size)
        self.lstm = nn.LSTM(input_size=pool_size * cnn_channels, hidden_size=lstm_hidden, num_layers=lstm_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(lstm_hidden, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input size (batch size, timesteps, features)
        B, T, F = x.shape
        x = x.view(B, 1, T, F) # Add channel dimension

        # Convolutional layers
        output = self.relu(self.conv1(x))
        output = self.relu(self.conv2(output))

        # Pooling
        output = self.pool(output) # Output size (batch size, channels, pool_size, pool_size)
        B, C, H, W = output.shape
        output = output.view(B, H, W * C) # Output size (batch size, pool_size (aka sequence length), lstm_features)

        # LSTM
        output, _ = self.lstm(output)
        output = self.linear(output)
        return output


class AHR_ConvEncoder(nn.Module):
    def __init__(self, num_features=1, num_classes=1):
        super(AHR_ConvEncoder, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_transpose = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        # Pooling layer
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)  # Global Average Pooling

        # Fully connected layer
        self.fc = nn.Linear(64, 256)

        # Final classification layer
        self.classification = nn.Linear(256, num_classes)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self.initialize_weights()


    def forward(self, x):
        out = x.permute(0, 2, 1)  # Change shape from (batch, seq_len, features) to (batch, features, seq_len)
        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        # Upsample using convoltional transpose layer
        out = self.conv_transpose(out)
        out = self.relu(out)

        out = self.global_max_pool(out)  # Output shape: (batch, 64, 1)
        out = out.squeeze(-1)  # Remove the last dimension -> (batch, 64)

        features = self.fc(out)  # Fully connected layer

        out = self.classification(features)  # Final output
        preds = self.sigmoid(out)
        return preds, features


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)


class AHR_LSTMEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float, num_features=1, num_classes=1):
        super(AHR_LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 256)
        self.classification = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.initialize_weights()

    def forward(self, x):
        out, _ = self.lstm(x)

        # Learned features for second-stage training
        features = self.fc(out[:, -1, :])

        out = self.classification(features)

        preds = self.sigmoid(out)

        return preds, features

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
