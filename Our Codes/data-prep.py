import os
import cv2
import json
import torch
from ultralytics import YOLO

"""This code uses Ultralytics YOLOv8 model (https://github.com/ultralytics/ultralytics)
and OpenCV for image annotation and label creation."""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8x.pt').to(device)
print(f"Using device: {device}")

rootDIR = os.path.dirname(__file__)
datasetPATH = os.path.join(rootDIR, '../Final-Dataset')
splits = ['train', 'val', 'test']

def getCOCO():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(model.names)]
    }

imgID = 0
ann_id = 0

for split in splits:
    splitDIR = os.path.join(datasetPATH, split)
    sharpDIR = os.path.join(splitDIR, 'sharp')    
    labelDIR = os.path.join(splitDIR, 'label')    
    gtDIR = os.path.join(splitDIR, 'ground_truth') 

    os.makedirs(labelDIR, exist_ok=True)
    os.makedirs(gtDIR, exist_ok=True)

    print(f"Processing split: {split}")
    imageFILES = sorted([f for f in os.listdir(sharpDIR) if f.endswith(('.jpg', '.png', '.jpeg'))])


    COCO = getCOCO()

    for imageNAME in imageFILES:
        try:
            inputPATH = os.path.join(sharpDIR, imageNAME)
            outputPATH = os.path.join(labelDIR, imageNAME)

            image = cv2.imread(inputPATH)
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            outputPRED = model(imageRGB, conf=0.1)

            height, width = image.shape[:2]

            COCO["images"].append({
                "id": imgID,
                "file_name": imageNAME,
                "width": width,
                "height": height
            })

            for pred in outputPRED:
                boxes = pred.boxes.xyxy.cpu().numpy()
                scores = pred.boxes.conf.cpu().numpy()
                labels = pred.boxes.cls.cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = map(float, box)
                    w = x2 - x1
                    h = y2 - y1

                    labelNAME = model.names[int(label)]
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, f"{labelNAME} {score:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    COCO["annotations"].append({
                        "id": ann_id,
                        "image_id": imgID,
                        "category_id": int(label),
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    ann_id += 1

            cv2.imwrite(outputPATH, image)
            print(f"Annotated: {imageNAME}")
            imgID += 1

        except Exception as e:
            print(f"Error processing {imageNAME}: {e}")

    cocoJSON = os.path.join(gtDIR, f"{split}.json")
    with open(cocoJSON, 'w') as f:
        json.dump(COCO, f)
    print(f"Saved COCO JSON: {cocoJSON}")

print("All splits processed successfully!")
