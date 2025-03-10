# data_preparation.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GrainEvolutionDataset(Dataset):
    def __init__(self, data, time_history, time_future, skip_steps, train=True, train_split=0.8, filter_tolerance=1e-6):
        super().__init__()
        self.time_history = time_history
        self.time_future = time_future
        self.skip_steps = skip_steps

        split_idx = int(train_split * data.shape[0])
        self.data = data[:split_idx] if train else data[split_idx:]
        self.sequences = self.create_sequences(self.data, filter_tolerance)

        print(f"{'Training' if train else 'Validation'} dataset: {len(self.sequences)} sequences")

    def create_sequences(self, data, filter_tolerance):
        sequences = []
        total_steps = self.time_history + self.skip_steps + self.time_future
        for sample in data:
            for t in range(sample.shape[0] - total_steps + 1):
                x = sample[t : t + self.time_history]
                y = sample[t + self.time_history + self.skip_steps : t + total_steps]
                if np.any(np.mean(np.abs(y - x[-1][np.newaxis, ...]), axis=(1, 2)) > filter_tolerance):
                    sequences.append((torch.tensor(x).float(), torch.tensor(y).float()))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def load_data(batch_size, time_history, time_future, skip_steps, filter_tolerance=1e-6, train_split=0.8):
    print("Loading data...")
    ss = np.load('Dataset_64.npy')
    ff = np.load('Dataset_b_64.npy')[:, 4:, :, :]
    data = np.concatenate((ss, ff), axis=0)[:, ::2, :, :]
    print(f"Data shape: {data.shape}")

    train_dataset = GrainEvolutionDataset(data, time_history, time_future, skip_steps, True, train_split, filter_tolerance)
    val_dataset = GrainEvolutionDataset(data, time_history, time_future, skip_steps, False, train_split, filter_tolerance)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset

class TestDataset(Dataset):
    def __init__(self, data, time_history, time_future, skip_steps, sample_idx=None):
        super().__init__()
        if sample_idx is not None:
            data = data[sample_idx:sample_idx+1]
        self.sequences = self.create_sequences(data, time_history, time_future, skip_steps)

    def create_sequences(self, data, time_history, time_future, skip_steps):
        sequences = []
        total_steps = time_history + skip_steps + time_future
        for sample in data:
            for t in range(sample.shape[0] - total_steps + 1):
                x = sample[t : t + time_history]
                y = sample[t + time_history + skip_steps : t + total_steps]
                sequences.append((torch.tensor(x).float(), torch.tensor(y).float()))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
