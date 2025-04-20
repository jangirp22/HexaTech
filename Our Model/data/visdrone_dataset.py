from torch.utils.data import Dataset
import cv2, os
import torch
from torchvision import transforms

class VisDroneDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.imgs = sorted([os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.endswith('.png')])
        self.transform = transform or transforms.ToTensor()

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)
