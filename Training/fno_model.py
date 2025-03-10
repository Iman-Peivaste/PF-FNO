# fno_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.weights1 = nn.Parameter(torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.weights1.shape[1], x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class FNO2d(nn.Module):
    def __init__(self, modes, width, time_history, time_future):
        super().__init__()
        self.fc0 = nn.Linear(time_history, width)
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, time_future)

    def forward(self, x):
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for conv, w in zip(self.convs, self.ws):
            x = x + w(F.gelu(conv(x)))
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = F.gelu(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        return x
