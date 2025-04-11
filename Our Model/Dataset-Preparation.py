import os
import cv2
import time
import torch
from ultralytics import YOLO

# We are utilizing YoLoV8 model for dataset preparation.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8x.pt').to(device)
print(f"Using device: {device}")

rootDIR = os.path.dirname(__file__)
datasetPATH = os.path.join(rootDIR, '../Dataset')
datasets = [folder for folder in os.listdir(datasetPATH) if os.path.isdir(os.path.join(datasetPATH, folder))]

for dataset in datasets:
    datasetDIR = os.path.join(datasetPATH, dataset)
    sharpDIR = os.path.join(datasetDIR, 'sharp')
    labelDIR = os.path.join(datasetDIR, 'label')

    os.makedirs(labelDIR, exist_ok=True)

    print(f"Processing dataset: {dataset}")
    
    try:
        imageFILE = [files for files in os.listdir(sharpDIR) if files.endswith(('.jpg', '.png', '.jpeg'))]
    except Exception as e:
        print(f"Failed to list files in {sharpDIR}: {e}")
        continue

    for imageNAME in imageFILE:
        try:
            inputPATH = os.path.join(sharpDIR, imageNAME)
            outputPATH = os.path.join(labelDIR, imageNAME)

            image = cv2.imread(inputPATH)
            if image is None:
                raise ValueError("Image not loaded... possibly corrupted or missing.")

            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            outputPRED = model(imageRGB, conf=0.1)

            for pred in outputPRED:
                boxes = pred.boxes.xyxy.cpu().numpy()
                scores = pred.boxes.conf.cpu().numpy()
                labels = pred.boxes.cls.cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = map(int, box)
                    label_name = model.names[int(label)]

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{label_name} {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imwrite(outputPATH, image)
            print(f"Saved label image: {outputPATH}")

        except Exception as e:
            print(f"Error processing image {imageNAME} in {dataset}: {e}")
