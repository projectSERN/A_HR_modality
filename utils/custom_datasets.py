import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, sequences, lengths):
        super().__init__()
        self.sequences = sequences
        self.lengths = lengths

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index], self.lengths[index]
    

def collate_seq_fn(batch):
    # Unpack sequences and lengths from the batch
    sequences, lengths = zip(*batch)

    # Convert to tensors
    lengths = torch.tensor(lengths)

    # Sort by lengths in descending order
    lengths, sorted_indices = lengths.sort(descending=True)
    sequences = [sequences[i] for i in sorted_indices]

    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # Split into features and target for loss calculation during training
    features = padded_sequences[:, :, :-1]
    target = padded_sequences[:, :, -1].unsqueeze(2)

    # Scale
    return features, target, lengths


def collate_encoder_fn(batch, device):
    features, labels = zip(*batch)
    
    # Convert to tensors
    features = [torch.tensor(f, dtype=torch.float32, device=device) for f in features]
    labels = torch.tensor(labels, dtype=torch.float32, device=device)
    
    # Pad sequences (batch_size, max_seq_len, num_features)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    
    return padded_features, labels