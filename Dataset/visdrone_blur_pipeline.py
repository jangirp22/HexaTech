# === VisDrone Blur Generation Pipeline ===
# Combines Trajectory, PSF, and BlurImage to generate blurred VisDrone dataset

import os
import numpy as np
from tqdm import tqdm
import cv2
from generate_trajectory import Trajectory
from generate_PSF import PSF
from blur_image import BlurImage

# === Configuration ===

# Input dataset folders (original VisDrone images)
folders = [
    'datasets/VisDrone-2019-DET/VisDrone2019-DET-train/images',
    'datasets/VisDrone-2019-DET/VisDrone2019-DET-val/images',
    'datasets/VisDrone-2019-DET/VisDrone2019-DET-test-dev/images',
]

# Output folders to save blurred images
folder_to_saves = [
    'datasets/VisDrone-2019-DET-blur/blurred-train/images',
    'datasets/VisDrone-2019-DET-blur/blurred-val/images',
    'datasets/VisDrone-2019-DET-blur/blurred-test/images',
]

# Parameters for trajectory variation (explosiveness)
explosion_values = [0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002]

# PSF part options (fraction of motion considered in blur kernel)
psf_parts = [1, 2, 3]

# Maximum number of retries per image
MAX_RETRIES = 5

# === Blur Generation Loop ===

def generate_blurred_dataset():
    for index in range(len(folders)):
        input_folder = folders[index]
        output_folder = folder_to_saves[index]

        os.makedirs(output_folder, exist_ok=True)
        print(f"\nProcessing: {input_folder} -> {output_folder}")

        # Get list of all images
        all_images = os.listdir(input_folder)
        
        # Get list of already processed images
        existing_images = set(os.listdir(output_folder)) if os.path.exists(output_folder) else set()
        print(f"Found {len(existing_images)} already processed images")
        
        # Filter out already processed images
        images_to_process = [img for img in all_images if img not in existing_images]
        print(f"Processing {len(images_to_process)} new images")

        for image_filename in tqdm(images_to_process):
            image_path = os.path.join(input_folder, image_filename)
            
            retry_count = 0
            success = False
            
            while not success and retry_count < MAX_RETRIES:
                try:
                    # Generate random parameters for blur
                    expl = np.random.choice(explosion_values)
                    trajectory = Trajectory(canvas=64, max_len=100, expl=expl).fit()
                    psfs = PSF(canvas=64, trajectory=trajectory).fit()
                    part = np.random.choice(psf_parts)

                    # Apply blur to image
                    blurrer = BlurImage(image_path, PSFs=psfs, part=part, path__to_save=output_folder)
                    blurrer.blur_image(save=True)
                    success = True
                except Exception as e:
                    retry_count += 1
                    print(f"Retrying {image_filename} ({retry_count}/{MAX_RETRIES}) due to error: {str(e)}")
                    if retry_count >= MAX_RETRIES:
                        print(f"Failed to process {image_filename} after {MAX_RETRIES} attempts, skipping")
                    continue

if __name__ == '__main__':
    generate_blurred_dataset()
