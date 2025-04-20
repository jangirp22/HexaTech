import torch 
import torch.nn.functional as F
from utils.box_ops import bbox_iou

def simota_matching(pred_boxes, pred_cls_logits, gt_boxes, gt_labels, num_classes, top_k=10, alpha=1.0, beta=6.0): 
    """ pred_boxes: [N, 4] predicted boxes (x1, y1, x2, y2) pred_cls_logits: [N, C] classification logits gt_boxes: [M, 4] ground-truth boxes gt_labels: [M] ground-truth labels """ 
    N, C = pred_cls_logits.shape 
    M = gt_boxes.shape[0]

    ious = bbox_iou(pred_boxes, gt_boxes)  # shape [N, M]

    cls_probs = F.softmax(pred_cls_logits, dim=-1)  # shape [N, C]
    cls_cost = -cls_probs[:, gt_labels]  # shape [N, M]

    total_cost = alpha * cls_cost + beta * (1.0 - ious)

    matching_matrix = torch.zeros_like(total_cost)
    for m in range(M):
        _, topk_idxs = torch.topk(total_cost[:, m], k=min(top_k, N), largest=False)
        matching_matrix[topk_idxs, m] = 1.0

    prior_match = matching_matrix.sum(dim=1)
    multiple_match_idxs = (prior_match > 1).nonzero(as_tuple=True)[0]
    for idx in multiple_match_idxs:
        costs = total_cost[idx]
        min_gt = torch.argmin(costs)
        matching_matrix[idx] = 0
        matching_matrix[idx, min_gt] = 1.0

    matched_gt_inds = matching_matrix.argmax(dim=1)
    fg_mask = matching_matrix.sum(dim=1) > 0

    return matched_gt_inds, fg_mask, matching_matrix
