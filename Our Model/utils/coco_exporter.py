import json

def save_coco(results, output_json):
    coco = {"images": [], "annotations": [], "categories": [],}
    ann_id = 1

    for i, res in enumerate(results):
        coco["images"].append({
            "id": i+1, "file_name": res['filename'],
            "width": res['width'], "height": res['height']
        })

        for box in res['boxes']:
            coco["annotations"].append({
                "id": ann_id, "image_id": i+1,
                "category_id": box['class_id'],
                "bbox": box['bbox'], "area": box['bbox'][2]*box['bbox'][3],
                "iscrowd": 0
            })
            ann_id += 1

    coco["categories"] = [{"id": i, "name": f"class_{i}"} for i in range(1,81)]
    with open(output_json, 'w') as f:
        json.dump(coco, f)
