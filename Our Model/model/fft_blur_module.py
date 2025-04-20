import torch
import torch.nn as nn
import torch.fft

class FFTBlur(nn.Module):
    def __init__(self, strength=0.5):
        super(FFTBlur, self).__init__()
        self.strength = strength

    def forward(self, img_tensor):
        B, C, H, W = img_tensor.shape
        img_fft = torch.fft.fft2(img_tensor)
        freq = torch.fft.fftshift(img_fft)

        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=img_tensor.device),
                                torch.linspace(-1, 1, W, device=img_tensor.device), indexing='ij')
        gauss = torch.exp(-(xx**2 + yy**2) / (2 * self.strength**2))
        gauss = gauss[None, None, :, :].repeat(B, C, 1, 1)

        blurred_freq = freq * gauss
        blurred_img = torch.real(torch.fft.ifft2(torch.fft.ifftshift(blurred_freq)))
        return blurred_img