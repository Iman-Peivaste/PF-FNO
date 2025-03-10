# tester.py
import torch
import numpy as np
from utils import load_trained_fno, visualize_results
import os

# Parameters (must match training)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_HISTORY = 5
TIME_FUTURE = 5
SKIP_STEPS = 5
MODES = 20
WIDTH = 32

def test_single_sample(model, resolution, sample_idx, device):
    test_data_path = f"Dataset2_{resolution}.npy"
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"File {test_data_path} not found.")
    
    data = np.load(test_data_path)[:, ::2, :, :]
    
    if sample_idx >= data.shape[0]:
        raise IndexError(f"sample_idx={sample_idx} out of bounds (max={data.shape[0]-1}).")
    
    sequence = data[sample_idx]
    total_required = TIME_HISTORY + SKIP_STEPS + TIME_FUTURE
    
    if sequence.shape[0] < total_required:
        raise ValueError("Insufficient frames for the required sequence length.")
    
    x_np = sequence[:TIME_HISTORY]
    y_true_np = sequence[TIME_HISTORY + SKIP_STEPS : total_required]

    x_torch = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        y_pred_torch = model(x_torch)
    y_pred_np = y_pred_torch.cpu().numpy().squeeze(0)

    visualize_results(x_np, y_true_np, y_pred_np, sample_idx, resolution)

if __name__ == "__main__":
    model = load_trained_fno(
        checkpoint_path="best_model.pt",
        modes=MODES,
        width=WIDTH,
        time_history=TIME_HISTORY,
        time_future=TIME_FUTURE,
        device=DEVICE
    )

    test_single_sample(
        model=model,
        resolution=512,  # Adjust as needed
        sample_idx=1,
        device=DEVICE
    )
