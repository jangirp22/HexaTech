import os
import cv2

# Path to your dataset
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Final-Dataset")
splits = ['train', 'val', 'test']
blur_threshold = 70.0  # You can change this threshold as needed

def compute_blur_score(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return float('inf')  # Treat unreadable images as non-blurry
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var

def process_split(split):
    split_path = os.path.join(root_dir, split)
    blur_dir = os.path.join(split_path, "blur")
    sharp_dir = os.path.join(split_path, "sharp")

    # Collect all image names
    image_files = sorted([
        f for f in os.listdir(blur_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    kept_images = []
    for image_name in image_files:
        blur_path = os.path.join(blur_dir, image_name)
        score = compute_blur_score(blur_path)

        if score < blur_threshold:
            kept_images.append(image_name)
        else:
            # Remove blur and corresponding sharp image
            sharp_path = os.path.join(sharp_dir, image_name)
            if os.path.exists(blur_path):
                os.remove(blur_path)
                print(f"Removed blurry image: {blur_path}")
            if os.path.exists(sharp_path):
                os.remove(sharp_path)
                print(f"Removed corresponding sharp image: {sharp_path}")

    # Now rename remaining images to 00000.png, 00001.png, ...
    for idx, old_name in enumerate(sorted(kept_images)):
        new_name = f"{idx:05d}.png"

        old_blur_path = os.path.join(blur_dir, old_name)
        old_sharp_path = os.path.join(sharp_dir, old_name)

        new_blur_path = os.path.join(blur_dir, new_name)
        new_sharp_path = os.path.join(sharp_dir, new_name)

        if os.path.exists(old_blur_path):
            os.rename(old_blur_path, new_blur_path)
        if os.path.exists(old_sharp_path):
            os.rename(old_sharp_path, new_sharp_path)

    print(f"Finished processing {split}: Kept {len(kept_images)} images")

if __name__ == "__main__":
    for split in splits:
        print(f"\nProcessing split: {split}")
        process_split(split)
    print("\nAll done!")
