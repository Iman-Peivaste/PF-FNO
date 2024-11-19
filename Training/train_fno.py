import os
import torch
import pickle
from timeit import default_timer
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from Adam import Adam
from utilities3 import LpLoss
from fno_model import FNO2D

if __name__ == "__main__":
    # Load preprocessed data
    train_a = torch.from_numpy(np.load("train_a.npy")).permute(0, 2, 3, 1).float()
    train_u = torch.from_numpy(np.load("train_u.npy")).permute(0, 2, 3, 1).float()
    test_a = torch.from_numpy(np.load("test_a.npy")).permute(0, 2, 3, 1).float()
    test_u = torch.from_numpy(np.load("test_u.npy")).permute(0, 2, 3, 1).float()

    # Data loaders
    batch_size = 20
    train_loader = DataLoader(TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

    # Model, optimizer, and scheduler
    modes, width, time_steps, epochs = 20, 40, 10, 300
    model = FNO2D(modes, modes, width, time_steps).to(torch.device("cuda"))
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    loss_fn = LpLoss(size_average=False)

    # Training loop
    train_losses, test_losses = [], []
    total_start_time = default_timer()

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        epoch_loss = 0
        for xx, yy in train_loader:
            xx, yy = xx.cuda(), yy.cuda()
            optimizer.zero_grad()

            loss = 0
            for t in range(0, yy.shape[-1], 1):
                y = yy[..., t:t + 1]
                im = model(xx)
                loss += loss_fn(im.reshape(xx.shape[0], -1), y.reshape(xx.shape[0], -1))
                xx = torch.cat((xx[..., 1:], im), dim=-1)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader.dataset))

        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            for xx, yy in test_loader:
                xx, yy = xx.cuda(), yy.cuda()
                loss = 0
                for t in range(0, yy.shape[-1], 1):
                    y = yy[..., t:t + 1]
                    im = model(xx)
                    loss += loss_fn(im.reshape(xx.shape[0], -1), y.reshape(xx.shape[0], -1))
                    xx = torch.cat((xx[..., 1:], im), dim=-1)
                epoch_loss += loss.item()
            test_losses.append(epoch_loss / len(test_loader.dataset))

        scheduler.step()
        print(f"Epoch {epoch + 1}: Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

    print(f"Training completed in {default_timer() - total_start_time:.2f} seconds")

    # Save model and results
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/FNO_model.pth")
    with open("train_losses.pkl", "wb") as f:
        pickle.dump(train_losses, f)
    with open("test_losses.pkl", "wb") as f:
        pickle.dump(test_losses, f)
