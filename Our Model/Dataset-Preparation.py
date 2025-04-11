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
    labelDIR = os.path.join(datasetDIR, 'label')  # Visual annotations
    gtDIR = os.path.join(datasetDIR, 'ground_truth')  # Structured .txt labels

    os.makedirs(labelDIR, exist_ok=True)
    os.makedirs(gtDIR, exist_ok=True)

    print(f"Processing dataset: {dataset}")
    imageFILE = [files for files in os.listdir(sharpDIR) if files.endswith(('.jpg', '.png', '.jpeg'))]

    for imageNAME in imageFILE:
        try:
            inputPATH = os.path.join(sharpDIR, imageNAME)
            outputPATH = os.path.join(labelDIR, imageNAME)
            label_txt_path = os.path.join(gtDIR, imageNAME.rsplit('.', 1)[0] + '.txt')

            image = cv2.imread(inputPATH)
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            outputPRED = model(imageRGB, conf=0.1)

            height, width = image.shape[:2]

            with open(label_txt_path, 'w') as f:
                for pred in outputPRED:
                    boxes = pred.boxes.xyxy.cpu().numpy()
                    scores = pred.boxes.conf.cpu().numpy()
                    labels = pred.boxes.cls.cpu().numpy()

                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = map(int, box)
                        labelNAME = model.names[int(label)]
                        classID = int(label)

                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, f"{labelNAME} {score:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        centerX = ((x1 + x2) / 2) / width
                        centerY = ((y1 + y2) / 2) / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height
                        f.write(f"{classID} {centerX:.6f} {centerY:.6f} {w:.6f} {h:.6f}\n")

            cv2.imwrite(outputPATH, image)
            print(f"Saved annotated image: {outputPATH}")
            print(f"Saved ground truth: {label_txt_path}")

        except Exception as e:
            print(f"Error processing {imageNAME}: {e}")
