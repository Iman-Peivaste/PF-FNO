import torch
import torch.nn.functional as F
from fno_model import FNO2D
from test_data_preparation import TestDataLoader
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model, test_loader, device):
    model.eval()
    predictions, ground_truths = [], []

    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            pred = []

            for _ in range(10):  # Assuming 10 time steps
                im = model(xx)
                pred.append(im.cpu())
                xx = torch.cat((xx[..., 1:], im), dim=-1)

            predictions.append(torch.cat(pred, dim=-1))
            ground_truths.append(yy.cpu())

    return torch.cat(predictions, dim=0), torch.cat(ground_truths, dim=0)


def visualize_sample(predictions, ground_truths, sample_idx):
    pred = predictions[sample_idx].numpy()
    gt = ground_truths[sample_idx].numpy()
    error = np.abs(gt - pred)

    for t in range(pred.shape[-1]):
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].imshow(gt[:, :, t], cmap='gray')
        axs[0].set_title(f"Ground Truth - Timestep {t + 1}")
        axs[1].imshow(pred[:, :, t], cmap='gray')
        axs[1].set_title(f"Prediction - Timestep {t + 1}")
        im = axs[2].imshow(error[:, :, t], cmap='hot')
        axs[2].set_title(f"Error - Timestep {t + 1}")
        plt.colorbar(im, ax=axs[2])
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "FNO_model.pth"
    modes1, modes2, width, time_steps = 16, 16, 16, 10

    # Load pre-trained model
    model = FNO2D.load_model(model_path, modes1, modes2, width, time_steps, device)

    # Load test data
    loader = TestDataLoader("Dataset_b_256.npy", 10, 10, 1)
    dataset = loader.load_and_preprocess()
    xx, yy = loader.create_sequences(dataset)
    xx, yy = loader.filter_identical_sequences(xx, yy)

    test_a = torch.from_numpy(xx).float().permute(0, 2, 3, 1)
    test_u = torch.from_numpy(yy).float().permute(0, 2, 3, 1)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=20)

    # Evaluate model
    predictions, ground_truths = evaluate_model(model, test_loader, device)

    # Compute metrics
    mse = F.mse_loss(predictions, ground_truths)
    mae = F.l1_loss(predictions, ground_truths)
    print(f"MSE: {mse.item():.4f}, MAE: {mae.item():.4f}")

    # Visualize results
    visualize_sample(predictions, ground_truths, sample_idx=0)
