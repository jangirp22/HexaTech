import torch
import torch.nn as nn

class YOLODetectionHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=10, num_anchors=1):
        super(YOLODetectionHead, self).__init__()
        self.num_classes = num_classes
        self.pred = nn.Conv2d(in_channels, num_anchors * (4 + 1 + num_classes), kernel_size=1)

    def forward(self, x):
        pred = self.pred(x)
        B, _, H, W = pred.shape
        pred = pred.view(B, 1, 4 + 1 + self.num_classes, H, W)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        return pred


# box_ops.py (in utils/)

