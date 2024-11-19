import numpy as np
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, dataset_path, shift_number, stride, time_steps):
        self.dataset_path = dataset_path
        self.shift_number = shift_number
        self.stride = stride
        self.time_steps = time_steps

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

    def remove_identical_sequences(self, xx, yy, tolerance=1e-6):
        difference = np.abs(xx - yy)
        equal_arrays = np.all(difference < tolerance, axis=(1, 2, 3))
        return xx[~equal_arrays], yy[~equal_arrays]


if __name__ == "__main__":
    dataset_path = "Dataset_b_64.npy"
    shift_number = 10
    time_steps = 10
    stride = 9

    data_loader = DataLoader(dataset_path, shift_number, stride, time_steps)
    dataset = data_loader.load_and_preprocess()
    xx, yy = data_loader.create_sequences(dataset)
    xx, yy = data_loader.remove_identical_sequences(xx, yy)

    #train_a, test_a, train_u, test_u = train_test_split(xx, yy, test_size=0.2, random_state=42)
    split_idx = int(len(xx) * 0.8)  # 80% for training, 20% for testing

  # Split the data
    train_a, test_a = xx[:split_idx], xx[split_idx:]  # First 80% for training, last 20% for testing
    train_u, test_u = yy[:split_idx], yy[split_idx:]  # Same for outputs
    

    np.save("train_a.npy", train_a)
    np.save("train_u.npy", train_u)
    np.save("test_a.npy", test_a)
    np.save("test_u.npy", test_u)

    print(f"Train and test datasets saved. Train shape: {train_a.shape}, Test shape: {test_a.shape}")
