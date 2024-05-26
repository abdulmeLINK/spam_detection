import numpy as np
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, max_length):
        self.texts = [self.pad_sequence(text, max_length) for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

    def pad_sequence(self, sequence, max_length):
        padded_sequence = np.zeros(max_length, dtype=int)
        padded_sequence[:len(sequence)] = sequence
        return padded_sequence
