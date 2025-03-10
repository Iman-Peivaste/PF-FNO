# trainer.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_preparation import load_data
from fno_model import FNO2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 20
LEARNING_RATE = 1e-3
EPOCHS = 200
MODES = 20
WIDTH = 32
TIME_HISTORY = 5
TIME_FUTURE = 5
SKIP_STEPS = 5

train_loader, val_loader, _, val_dataset = load_data(
    BATCH_SIZE, TIME_HISTORY, TIME_FUTURE, SKIP_STEPS
)

model = FNO2d(MODES, WIDTH, TIME_HISTORY, TIME_FUTURE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
criterion = nn.L1Loss()

for epoch in range(EPOCHS):
    model.train()
    train_loss = sum(
        criterion(model(x.to(device)), y.to(device)).item()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    ) / len(train_loader)

    model.eval()
    val_loss = sum(
        criterion(model(x.to(device)), y.to(device)).item()
        for x, y in val_loader
    ) / len(val_loader)

    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if epoch == 0 or val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')

model.load_state_dict(torch.load('best_model.pt'))

# Plotting Loss
plt.plot([train_loss], label='Train Loss')
plt.plot([val_loss], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
