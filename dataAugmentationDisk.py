import os
import json
import cv2
import albumentations as A
import random

# Paths
INPUT_DIR = './datasets/NewDatasetAug/newVal/'  # Folder with original images
ANNOTATIONS_FILE = './datasets/NewDatasetAug/newVal/val_annotations.json'  # VGG JSON file
OUTPUT_DIR = './datasets/NewDatasetAug/newVal/val/'  # Folder for augmented images
OUTPUT_ANNOTATIONS_FILE = 'augmented_val_annotations.json'  # Augmented annotations file
NUM_AUGMENTATIONS = 5  # How many augmentations per image

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load VGG annotations
with open(ANNOTATIONS_FILE) as f:
    annotations = json.load(f)

# New annotations dict for Mask R-CNN format
augmented_annotations = {}

# Define class mapping
class_map = {
    "broken part": 1,
    "crack": 2,
    "dent": 3,
    "lamp broken": 4,
    "missing part": 5,
    "scratch": 6,
    "bg": 0
}

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=45, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
], keypoint_params=A.KeypointParams(format='xy'))

annotation_id = 1
image_id = 1

# Loop through all images
for img_key, img_data in annotations.items():
    img_name = img_data['filename']
    img_path = os.path.join(INPUT_DIR, img_name)

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: couldn't read {img_path}")
        continue

    # Extract polygons (keypoints) from annotations
    regions = img_data['regions']
    if isinstance(regions, dict):
        regions = list(regions.values())

    region_data = []
    keypoints = []

    for region in regions:
        shape = region['shape_attributes']
        if shape['name'] == 'polygon':
            all_points_x = shape['all_points_x']
            all_points_y = shape['all_points_y']

            # Add to keypoints for albumentations
            keypoints.extend([(x, y) for x, y in zip(all_points_x, all_points_y)])

            # Create region entry for the original image
            region_data.append({
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": all_points_x,
                    "all_points_y": all_points_y
                },
                "region_attributes": {
                    "label": region['region_attributes'].get('label', 'bg')
                }
            })

    if not region_data:
        print(f"No valid regions found for {img_name}, skipping image.")
        continue

    # ---- Save original image ----
    original_img_name = f"{os.path.splitext(img_name)[0]}_original.jpg"
    original_img_path = os.path.join(OUTPUT_DIR, original_img_name)
    cv2.imwrite(original_img_path, image)

    # Create entry in the augmented annotations dict
    augmented_annotations[original_img_name + "0"] = {
        "filename": original_img_name,
        "size": 0,
        "regions": region_data
    }

    # Process augmentations
    for i in range(NUM_AUGMENTATIONS):
        # Apply augmentation
        augmented = transform(image=image, keypoints=keypoints)
        aug_image = augmented['image']
        aug_keypoints = augmented['keypoints']

        # Reconstruct the polygon for the augmented image
        aug_points_x = [int(x) for x, y in aug_keypoints]
        aug_points_y = [int(y) for x, y in aug_keypoints]

        # Save augmented image
        aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
        output_img_path = os.path.join(OUTPUT_DIR, aug_img_name)
        cv2.imwrite(output_img_path, aug_image)

        # Augmented region data (same as original)
        aug_region_data = []
        for region in region_data:
            aug_region_data.append({
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": aug_points_x,
                    "all_points_y": aug_points_y
                },
                "region_attributes": {
                    "label": region['region_attributes'].get('label', 'bg')
                }
            })

        # Save augmented annotations
        augmented_annotations[aug_img_name + "0"] = {
            "filename": aug_img_name,
            "size": 0,
            "regions": aug_region_data
        }

        image_id += 1

# Save new annotations in the correct format
with open(os.path.join(OUTPUT_DIR, OUTPUT_ANNOTATIONS_FILE), 'w') as f:
    json.dump(augmented_annotations, f, indent=2)

print("Augmentation completed and annotations saved in the correct format!")
