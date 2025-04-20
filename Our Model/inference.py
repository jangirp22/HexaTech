import torch
import os
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from torchvision import transforms as T
from torchvision.ops import nms
from PIL import Image
from data.coco_loader import CocoDetectionDataset
from model.fft_blur_module import FFTBlur
from model.detection_head import YOLODetectionHead
from model.deblurnet import DeblurNET # import your model
import torch.nn.functional as F
from utils.box_ops import box_xyxy_to_cxcywh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('models/model_epoch_6.pth')
num_classes = 10
dreb = DeblurNET(pretrained=True).to(device)
detector = YOLODetectionHead(in_channels=64, num_classes=10).to(device)

dreb.load_state_dict(checkpoint['dreb_state_dict'])
detector.load_state_dict(checkpoint['detector_state_dict'])

dreb.eval()
detector.eval()

fft = FFTBlur().to(device)
fft.eval()

for param in fft.parameters():
    param.requires_grad = False

json_path = './dataset/val/annotations.json'  # COCO annotations path
image_dir = './dataset/val/images'  # Directory containing the images
dataset = CocoDetectionDataset(json_path, image_dir)

data_loader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)

with torch.no_grad():  
    for images, targets in data_loader:
        images = images.to(device)
        dreb_outputs = dreb(images)

        detections = detector(dreb_outputs)
        B, A, H, W, _ = detections.shape
        for b in range(B):  
            pred = detections[b, 0]  # shape: (H, W, 5 + num_classes)  
            pred = pred.view(-1, 5 + num_classes)  

            boxes_pred = pred[:, :4]          # [N, 4]
            scores = pred[:, 4]               # objectness logits [N]
            cls_pred = pred[:, 5:]            # class logits [N, C]

            cls_probs = F.softmax(cls_pred, dim=-1)   
            obj_probs = torch.sigmoid(scores)          
            conf_scores, cls_labels = cls_probs.max(dim=-1)  
            final_scores = conf_scores * obj_probs     

            # Filter 1: only boxes with all positive coordinates
            positive_mask = (boxes_pred > 0).all(dim=1)  # shape: [N]
            boxes_pred = boxes_pred[positive_mask]
            final_scores = final_scores[positive_mask]
            cls_labels = cls_labels[positive_mask]

            # Filter 2: confidence threshold
            threshold = 0.055
            keep = final_scores > threshold
            boxes = boxes_pred[keep]
            scores = final_scores[keep]
            labels = cls_labels[keep]

            # Apply NMS
            # if boxes.size(0) > 0:
            #     keep_nms = nms(boxes, scores, iou_threshold=0.5)
            #     boxes = boxes[keep_nms]
            #     labels = labels[keep_nms]
            #     scores = scores[keep_nms]

            print(f'Detected {len(boxes)} objects')
            for box, label, score in zip(boxes, labels, scores):
                print(f'Box: {box}, Label: {label}, Score: {score}')

            gt_boxes, gt_labels = targets[b]

            print(f"Ground Truth for Image {b} = {len(gt_boxes)}:")
            for box, label in zip(gt_boxes, gt_labels):
                print(f"  GT Box: {box.tolist()} - Label: {label.item()}")
