import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_ops import bbox_iou
from model.simOTA_matcher import simota_matching

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=10, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_box = lambda_coord       # alias for consistency  
        self.lambda_obj = 1.0                # lamda_obj, lamda_cls can be used as trainable parameters.  
        self.lambda_cls = 1.0                

        self.smooth_l1 = nn.SmoothL1Loss()  
        self.bce = nn.BCEWithLogitsLoss()  
        self.ce = nn.CrossEntropyLoss()  

    def forward(self, preds, targets):  
        """  
        preds: (B, A, H, W, 5 + num_classes)  
        targets: list of tuples -> [(boxes_tensor, labels_tensor), ...]  
        """  
        box_loss, obj_loss, cls_loss, total_loss = 0, 0, 0, 0
        B = preds.size(0)
        for b in range(B):
            pred = preds[b, 0].view(-1, 5 + self.num_classes)  # [H*W, 5 + C]
            boxes_pred = pred[:, :4]
            obj_pred = pred[:, 4]
            cls_pred = pred[:, 5:]

            gt_boxes, gt_labels = targets[b]
            if len(gt_boxes) == 0:
                obj_loss += self.bce(obj_pred, torch.zeros_like(obj_pred))
                continue

            gt_boxes = gt_boxes.to(pred.device)
            gt_labels = gt_labels.to(pred.device)

            matched_gt_inds, fg_mask, matching_matrix = simota_matching(
                boxes_pred, cls_pred, gt_boxes, gt_labels, num_classes=self.num_classes
            )

            pred_boxes_fg = boxes_pred[fg_mask]
            obj_pred_fg = obj_pred[fg_mask]
            cls_pred_fg = cls_pred[fg_mask]
            assigned_gt_boxes = gt_boxes[matched_gt_inds[fg_mask]]
            assigned_gt_labels = gt_labels[matched_gt_inds[fg_mask]]

            # Box loss
            box_loss += self.smooth_l1(pred_boxes_fg, assigned_gt_boxes)

            # Objectness loss
            iou_targets = bbox_iou(pred_boxes_fg, assigned_gt_boxes).diag().detach()
            obj_pred_fg = torch.sigmoid(obj_pred_fg)
            obj_loss += self.bce(obj_pred_fg, iou_targets)

            # Classification loss
            cls_target = torch.zeros_like(cls_pred_fg)
            cls_target[range(len(assigned_gt_labels)), assigned_gt_labels] = 1.0
            cls_loss += F.binary_cross_entropy_with_logits(cls_pred_fg, cls_target) 

        total_loss = (self.lambda_box * box_loss + self.lambda_obj * obj_loss + self.lambda_cls * cls_loss) / B  
        return total_loss, box_loss / B, obj_loss / B, cls_loss / B
