# DrebNet with Frequency-Domain Enhancements

This project enhances the original DrebNet architecture by integrating recent advances in frequency-domain processing for image deblurring and restoration.
Inspired by the paper *"Intriguing Findings of Frequency Selection for Image Deblurring"* (AAAI 2023), 
we introduce novel modules that improve the model's robustness and detail preservation under multi-degradation scenarios.

---

## Key Features

- Res FFT-ReLU Blocks: Residual blocks combining spatial and frequency-domain representations.
- Learnable Frequency Thresholding: 1x1 convolutions learn to filter meaningful frequency components adaptively.
- Dual-Domain Learning: Spatial and frequency streams are processed and fused for rich feature extraction.
- Visual FFT Attention: Soft attention maps guide the model to high-frequency informative regions.
- Frequency-Aware Loss Function: Enhances detail reconstruction by penalizing frequency-domain mismatches.

---

## Architecture Overview

```
Input → DualDomainExtractor → Res FFT-ReLU → FFT Attention → Res FFT-ReLU → Decoder → Output
```

---

##  Setup

### Dependencies
- PyTorch
- TorchVision
- Python 3.8+

Install via pip:
```bash
pip install torch torchvision
```

##Dataset
Replace `YourDataset` in the code with your actual dataset class. Ensure it returns input-target pairs for deblurring.

---

##Training
```bash
python DrebNet_Main.py --epochs 20 --lr 1e-4 --batch_size 8 --device cuda
```

Optional arguments:
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--batch_size`: Batch size
- `--device`: Training device (`cuda` or `cpu`)

---

## 📈 Loss Function
The loss used is:
```python
L = L1(output, target) + α × Frequency_L1(FFT(output), FFT(target))
```
# Each training epoch prints the average combined loss. You can also modify the script to visualize attention maps or FFT spectrums of outputs.


# This implementation draws inspiration from the DrebNet architecture and recent work in dual-domain learning for vision tasks.


# -------------------- Model and Training Code --------------------

import torch
import torch.nn as nn
import torch.fft
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from your_dataset import YourDataset  # replace with actual dataset module


class ResFFTReLUBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_path = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.freq_path_1x1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, x):
        x_spatial = self.spatial_path(x)
        x_fft = torch.fft.rfft2(x, norm='ortho')
        real, imag = x_fft.real, x_fft.imag
        fft_cat = torch.cat([real, imag], dim=1)
        fft_feat = self.freq_path_1x1(fft_cat)
        fft_real, fft_imag = fft_feat.chunk(2, dim=1)
        fft_complex = torch.complex(fft_real, fft_imag)
        x_ifft = torch.fft.irfft2(fft_complex, s=x.shape[-2:], norm='ortho')
        return x + x_spatial + x_ifft


class FFTAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        fft = torch.fft.fft2(x)
        magnitude = torch.abs(fft.real)
        attention = torch.sigmoid(self.conv(magnitude))
        return x * attention


class DualDomainExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.freq = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, x):
        x_spatial = self.spatial(x)
        x_fft = torch.fft.rfft2(x, norm='ortho')
        real, imag = x_fft.real, x_fft.imag
        fft_cat = torch.cat([real, imag], dim=1)
        freq_feat = self.freq(fft_cat)
        fft_real, fft_imag = freq_feat.chunk(2, dim=1)
        fft_complex = torch.complex(fft_real, fft_imag)
        x_ifft = torch.fft.irfft2(fft_complex, s=x.shape[-2:], norm='ortho')
        return x_spatial + x_ifft


class DrebNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            DualDomainExtractor(in_channels),
            ResFFTReLUBlock(base_channels),
            FFTAttention(base_channels),
            ResFFTReLUBlock(base_channels),
        )
        self.decoder = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def frequency_loss(pred, target):
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    return torch.mean(torch.abs(pred_fft - target_fft))


def combined_loss(pred, target, alpha=0.01):
    l1 = nn.L1Loss()(pred, target)
    freq = frequency_loss(pred, target)
    return l1 + alpha * freq


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = combined_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = DrebNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataset = YourDataset(transform=transforms.ToTensor())  # Replace with actual dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        epoch_loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")


if __name__ == '__main__':
    main()
