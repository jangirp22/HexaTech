import os

# Top 10 YOLOv8 COCO class indices
top10_class_ids = {0, 1, 2, 3, 5, 7, 15, 16, 56, 62}

# Automatically detect the directory where this script is located
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Dataset")

# Expected folder names inside each dataset folder
subfolders = ["sharp", "label", "ground_truth", "blur"]

def process_ground_truths():
    for dataset in os.listdir(root_dir):
        print(f"Processing: {dataset}")
        dataset_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        gt_folder = os.path.join(dataset_path, "ground_truth")
        if not os.path.isdir(gt_folder):
            continue

        for filename in os.listdir(gt_folder):
            if not filename.endswith(".txt"):
                continue

            txt_path = os.path.join(gt_folder, filename)

            # Read lines and filter based on class_id
            with open(txt_path, "r") as f:
                lines = f.readlines()

            filtered_lines = [
                line for line in lines
                if line.strip() and int(line.split()[0]) in top10_class_ids
            ]

            if filtered_lines:
                # Write back only the valid lines
                with open(txt_path, "w") as f:
                    f.writelines(filtered_lines)
            else:
                # If no valid lines left, delete the corresponding image files
                print(f"Removing empty annotation and associated files for: {filename}")

                base_name = os.path.splitext(filename)[0]

                for folder in subfolders:
                    folder_path = os.path.join(dataset_path, folder)
                    for ext in [".jpg", ".png"]:
                        file_path = os.path.join(folder_path, base_name + ext)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Deleted: {file_path}")

                if os.path.exists(txt_path):
                    os.remove(txt_path)
                    print(f"Deleted annotation file: {txt_path}")

if __name__ == "__main__":
    process_ground_truths()
