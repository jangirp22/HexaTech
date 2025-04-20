import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.deblurnet import DeblurNET
from model.detection_head import YOLODetectionHead
from model.yolo_loss import YOLOLoss
from model.fft_blur_module import FFTBlur
from data.coco_loader import CocoDetectionDataset

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    fft = FFTBlur().to(device)
    dreb = DeblurNET(pretrained=True).to(device)
    detector = YOLODetectionHead(in_channels=64, num_classes=10).to(device)
    criterion = YOLOLoss(num_classes=10)

    params = list(dreb.parameters()) + list(detector.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)

    dataset = CocoDetectionDataset(json_path='dataset/train/annotations.json', image_dir='dataset/train/images')
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)

    for epoch in range(50):
        dreb.train()
        total_loss = 0
        for i, (images, targets) in enumerate(loader):
            images = images.to(device)
            targets = [(b.to(device), l.to(device)) for b, l in targets]

            optimizer.zero_grad()

            blurred = fft(images)
            features = dreb(blurred)  # shape: (B, C, H, W)
            preds = detector(features)

            loss, box_loss, obj_loss, cls_loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch} Batch {i} - Total: {loss.item():.4f}, Box: {box_loss.item():.4f}, Obj: {obj_loss.item():.4f}, Cls: {cls_loss.item():.4f}")

        print(f"Epoch {epoch} completed. Avg loss: {total_loss / len(loader):.4f}")

        os.makedirs('models', exist_ok=True)  # <-- Ensure directory exists
        torch.save({
            'epoch': epoch,
            'dreb_state_dict': dreb.state_dict(),
            'detector_state_dict': detector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'models/model_epoch_{epoch}.pth')
        print(f"Model for epoch {epoch} saved!!!")

if __name__ == '__main__':
    run()
