import os
import json
import cv2

# INPUTS
VGG_ANNOTATIONS_FILE = './datasets/coco/val/vgg_annotations.json'  # VGG input file
IMAGES_DIR = './datasets/coco/val/'  # Folder where all images are
OUTPUT_COCO_FILE = './datasets/coco/val/coco_annotations.json'  # Output COCO file

# Class Names (7 classes including background)
CATEGORY_NAMES = [
    "bg",               # 0 - Background
    "broken part",      # 1
    "crack",            # 2
    "dent",             # 3
    "lamp broken",      # 4
    "missing part",     # 5
    "scratch"           # 6
]

# INIT COCO STRUCTURE
coco = {
    "images": [],
    "annotations": [],
    "categories": [{"id": i, "name": CATEGORY_NAMES[i]} for i in range(len(CATEGORY_NAMES))]
}

annotation_id = 1
image_id = 1

# Load VGG annotations
with open(VGG_ANNOTATIONS_FILE) as f:
    vgg_data = json.load(f)

# Process each image
for img_key, img_data in vgg_data.items():
    filename = img_data['filename']
    img_path = os.path.join(IMAGES_DIR, filename)

    # Read image size
    if not os.path.exists(img_path):
        print(f"Warning: {filename} not found, skipping...")
        continue
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Add image info
    coco['images'].append({
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    # Parse regions
    regions = img_data['regions']
    if isinstance(regions, dict):  # Sometimes regions is a dict
        regions = list(regions.values())

    for region in regions:
        shape_attr = region['shape_attributes']

        if shape_attr['name'] == 'polygon':
            all_points_x = shape_attr.get('all_points_x', [])
            all_points_y = shape_attr.get('all_points_y', [])

            # Skip if points are empty
            if not all_points_x or not all_points_y:
                print(f"Warning: No points found in {filename} for region, skipping...")
                continue

            segmentation = []

            # Build segmentation
            for x, y in zip(all_points_x, all_points_y):
                segmentation.extend([x, y])

            # Compute bounding box
            xmin = min(all_points_x)
            xmax = max(all_points_x)
            ymin = min(all_points_y)
            ymax = max(all_points_y)
            width_box = xmax - xmin
            height_box = ymax - ymin
            area = width_box * height_box

            # Determine category_id (class)
            # Assuming each region has a "region_attributes" field with "class" info.
            region_class = region['region_attributes'].get('class', None)

            if region_class:
                try:
                    # Find the class index (ID)
                    category_id = CATEGORY_NAMES.index(region_class)
                except ValueError:
                    print(f"Warning: Class '{region_class}' not found in CATEGORY_NAMES for {filename}, skipping region...")
                    continue
            else:
                print(f"Warning: No class found for region in {filename}, skipping...")
                continue

            # Add annotation for the class
            coco['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [segmentation],
                "bbox": [xmin, ymin, width_box, height_box],
                "area": area,
                "iscrowd": 0
            })

            annotation_id += 1

    image_id += 1

# Save COCO JSON
with open(OUTPUT_COCO_FILE, 'w') as f:
    json.dump(coco, f, indent=2)

print(f"✅ COCO annotations saved to {OUTPUT_COCO_FILE}")
