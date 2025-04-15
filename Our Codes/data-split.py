import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm

BASE_DATASET_DIR = "../Dataset"
OUTPUT_DIR = "../Final-Dataset"
IMG_SIZE = (512, 512)
SEED = 42
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}
IMAGE_TYPES = ['blur', 'sharp']

resizeTRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
])

def makeDIRS():
    for split in ['train', 'val', 'test']:
        for typ in IMAGE_TYPES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, typ), exist_ok=True)

def resizeANDsave(src, dst):
    img = Image.open(src).convert("RGB")
    img = resizeTRANSFORM(img)
    img.save(dst)

def collectImagePairs():
    ALL_PAIRS = []

    for datasetNAME in os.listdir(BASE_DATASET_DIR):
        datasetPATH = os.path.join(BASE_DATASET_DIR, datasetNAME)
        blurDIR = os.path.join(datasetPATH, 'blur')
        sharpDIR = os.path.join(datasetPATH, 'sharp')

        if not os.path.isdir(blurDIR) or not os.path.isdir(sharpDIR):
            continue

        blurFILES = sorted(os.listdir(blurDIR))
        sharpFILES = sorted(os.listdir(sharpDIR))

        min_len = min(len(blurFILES), len(sharpFILES))
        print(f"{datasetNAME} has {min_len} valid image pairs...")

        for i in range(min_len):
            blur_path = os.path.join(blurDIR, blurFILES[i])
            sharp_path = os.path.join(sharpDIR, sharpFILES[i])
            ALL_PAIRS.append((blur_path, sharp_path))

    return ALL_PAIRS

def prepareDATA():
    makeDIRS()
    imgPAIRS = collectImagePairs()
    print(f"Total image pairs found: {len(imgPAIRS)}")

    trainPAIRS, tempPAIRS = train_test_split(imgPAIRS, test_size=1 - SPLIT_RATIOS["train"], random_state=SEED)
    valPAIRS, testPAIRS = train_test_split(tempPAIRS, test_size=SPLIT_RATIOS["test"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]), random_state=SEED)

    splits = {
        "train": trainPAIRS,
        "val": valPAIRS,
        "test": testPAIRS
    }

    for split, pairs in splits.items():
        print(f"\nProcessing split: {split}")
        for idx, (blurPATH, sharpPATH) in tqdm(enumerate(pairs), total=len(pairs), desc=f"{split} set"):
            img_name = f"{idx:05d}.png"
            resizeANDsave(blurPATH, os.path.join(OUTPUT_DIR, split, 'blur', img_name))
            resizeANDsave(sharpPATH, os.path.join(OUTPUT_DIR, split, 'sharp', img_name))
            
    print("Dataset resize and split complete!!!")

if __name__ == "__main__":
    prepareDATA()
