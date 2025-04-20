import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image

class CocoDetectionDataset(Dataset):
    def __init__(self, json_path, image_dir):
        self.coco = COCO(json_path)
        self.image_dir = image_dir
        self.ids = list(self.coco.imgs.keys())
        self.tf = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            x1, y1, w, h = bbox
            boxes.append([x1 / 512, y1 / 512, w / 512, h / 512])
            labels.append(ann['category_id'] - 1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        path = os.path.join(self.image_dir, self.coco.loadImgs(img_id)[0]['file_name'])
        image = self.tf(Image.open(path).convert('RGB'))

        return image, (boxes, labels)

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        return images, targets
