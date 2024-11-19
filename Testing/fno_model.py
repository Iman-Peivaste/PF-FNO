import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2dFast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.weights1.shape[2], :self.weights1.shape[3]] = self.compl_mul2d(x_ft[:, :, :self.weights1.shape[2], :self.weights1.shape[3]], self.weights1)
        out_ft[:, :, -self.weights1.shape[2]:, :self.weights1.shape[3]] = self.compl_mul2d(x_ft[:, :, -self.weights1.shape[2]:, :self.weights1.shape[3]], self.weights2)
        return torch.fft.irfft2(out_ft)


class FNO2D(nn.Module):
    def __init__(self, modes1, modes2, width, time_steps):
        super().__init__()
        self.fc0 = nn.Linear(time_steps + 2, width)
        self.conv_layers = nn.ModuleList([SpectralConv2dFast(width, width, modes1, modes2) for _ in range(4)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)

        for conv, w in zip(self.conv_layers, self.w_layers):
            x = F.gelu(conv(x) + w(x))

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

    @staticmethod
    def get_grid(shape, device):
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).view(1, size_x, 1, 1).repeat(batch_size, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, device=device).view(1, 1, size_y, 1).repeat(batch_size, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    @staticmethod
    def load_model(model_path, modes1, modes2, width, time_steps, device):
        model = FNO2D(modes1, modes2, width, time_steps).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
