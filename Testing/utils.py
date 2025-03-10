# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from fno_model import FNO2d

def load_trained_fno(checkpoint_path, modes, width, time_history, time_future, device):
    model = FNO2d(modes, width, time_history, time_future).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def visualize_results(x_np, y_true_np, y_pred_np, sample_idx, resolution, output_dir="test_predictions"):
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Initial frames
    fig, axes = plt.subplots(1, x_np.shape[0], figsize=(3*x_np.shape[0], 3))
    for i, ax in enumerate(axes):
        im = ax.imshow(x_np[i])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$\eta$')
        ax.set_title(f"Initial frame {i+1}")
        ax.axis('off')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample{sample_idx}_initial.jpg"), dpi=500)
    plt.close(fig)

    # Figure 2: GT vs Prediction vs Error
    fig, axes = plt.subplots(3, y_true_np.shape[0], figsize=(3*y_true_np.shape[0], 9))

    for i in range(y_true_np.shape[0]):
        axes[0, i].imshow(y_true_np[i])
        axes[0, i].set_title(f"GT {i+1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(y_pred_np[i])
        axes[1, i].set_title(f"Pred {i+1}")
        axes[1, i].axis('off')

        err = y_pred_np[i] - y_true_np[i]
        im_err = axes[2, i].imshow(err, cmap='jet')
        plt.colorbar(im_err, ax=axes[2, i], fraction=0.046, pad=0.04)
        axes[2, i].set_title(f"Error {i+1}")
        axes[2, i].axis('off')

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample{sample_idx}_comparison.jpg"), dpi=500)
    plt.close(fig)
