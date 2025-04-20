import torch

def bbox_iou(boxes1, boxes2, eps=1e-7):
    """
    Compute pairwise IoU between two sets of boxes
    boxes1: [N, 4] in (x1, y1, x2, y2)
    boxes2: [M, 4] in (x1, y1, x2, y2)
    returns: [N, M] IoU matrix
    """
    N = boxes1.size(0)
    M = boxes2.size(0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]) *
             (boxes1[:, 3] - boxes1[:, 1])).unsqueeze(1)  # [N, 1]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]) *
             (boxes2[:, 3] - boxes2[:, 1])).unsqueeze(0)  # [1, M]

    union = area1 + area2 - inter + eps
    iou = inter / union
    return iou  # [N, M]


def bbox_giou(box1, box2):
    iou = bbox_iou(box1, box2)
    x1 = torch.min(box1[:, 0], box2[:, 0])
    y1 = torch.min(box1[:, 1], box2[:, 1])
    x2 = torch.max(box1[:, 2], box2[:, 2])
    y2 = torch.max(box1[:, 3], box2[:, 3])
    c = (x2 - x1) * (y2 - y1)
    return iou - (c - (iou * (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]))) / c



def box_xyxy_to_cxcywh(boxes, img_width = 512, img_height=512):
    x1, y1, x2, y2 = boxes.unbind(-1)

    x1 = x1 * img_width
    y1 = y1 * img_height
    x2 = x2 * img_width
    y2 = y2 * img_height

    return torch.stack([x1, y1, x2, y2], dim=-1)

