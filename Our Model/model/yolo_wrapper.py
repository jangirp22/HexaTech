from ultralytics import YOLO
import numpy as np
import torch

class YOLOv8Wrapper:
    def __init__(self, weights='yolov8n.pt'):
        self.model = YOLO(weights)

    def detect(self, img_tensor):
        img_np = (img_tensor.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        results = self.model.predict(img_np)
        return results[0]
