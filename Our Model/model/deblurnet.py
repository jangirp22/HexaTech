import torch
import torch.nn as nn
import torchvision.models as models

class DeblurNET(nn.Module):
    def __init__(self, pretrained=True):
        super(DeblurNET, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        # Encoder: use pretrained ResNet up to layer3
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

        # Decoder: upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)  # Shape: (B, 64, H/8, W/8)
        return x
