from torch import nn

class AHREncoder(nn.Module):
    def __init__(self, num_features=1, num_classes=1):
        super(AHREncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)  # Global Average Pooling
        self.classification = nn.Linear(16, num_classes)  # Final classification/regression layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape from (batch, seq_len, features) to (batch, features, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.global_avg_pool(x)  # Output shape: (batch, 64, 1)
        x = x.squeeze(-1)  # Remove the last dimension -> (batch, 64)
        x = self.classification(x)  # Final output
        return x
