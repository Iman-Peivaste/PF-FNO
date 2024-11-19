import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class TestDataLoader:
    def __init__(self, dataset_path, shift_number, time_steps, stride):
        self.dataset_path = dataset_path
        self.shift_number = shift_number
        self.time_steps = time_steps
        self.stride = stride

    def load_and_preprocess(self):
        dataset = np.load(self.dataset_path)
        #dataset = dataset[:, 2:141, :, :][:, ::2, :, :]
        return dataset

    def create_sequences(self, dataset):
        inp = dataset[:, :dataset.shape[1] - self.shift_number, :, :]
        out = dataset[:, self.shift_number:dataset.shape[1], :, :]
        xx_list, yy_list = [], []

        for sample_idx in range(inp.shape[0]):
            for seq_idx in range((inp.shape[1] - self.time_steps) // self.stride + 1):
                start = seq_idx * self.stride
                end = start + self.time_steps
                xx_list.append(inp[sample_idx, start:end, :, :])
                yy_list.append(out[sample_idx, start:end, :, :])

        xx = np.stack(xx_list, axis=0)
        yy = np.stack(yy_list, axis=0)
        return xx, yy

    def filter_identical_sequences(self, xx, yy, tolerance=1e-6):
        difference = np.abs(xx - yy)
        equal_arrays = np.all(difference < tolerance, axis=(1, 2, 3))
        return xx[~equal_arrays], yy[~equal_arrays]


if __name__ == "__main__":
    dataset_path = "Dataset_b_256.npy"
    shift_number = 10
    time_steps = 10
    stride = 1

    loader = TestDataLoader(dataset_path, shift_number, time_steps, stride)
    dataset = loader.load_and_preprocess()
    xx, yy = loader.create_sequences(dataset)
    xx, yy = loader.filter_identical_sequences(xx, yy)

    test_a = torch.from_numpy(xx).float().permute(0, 2, 3, 1)
    test_u = torch.from_numpy(yy).float().permute(0, 2, 3, 1)

    batch_size = 20
    test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

    print(f"Test data prepared. Input shape: {test_a.shape}, Output shape: {test_u.shape}")
