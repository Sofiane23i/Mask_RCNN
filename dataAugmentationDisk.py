import os
import json
import cv2
import albumentations as A

# Paths
INPUT_DIR = './datasets/coco/train/'
ANNOTATIONS_FILE = './datasets/coco/train/vgg_annotations.json'
OUTPUT_DIR = './datasets/coco/train/newtrain/'
OUTPUT_ANNOTATIONS_FILE = 'vgg_annotations.json'
NUM_AUGMENTATIONS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ANNOTATIONS_FILE) as f:
    annotations = json.load(f)

augmented_annotations = {}

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
                       border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=0.5),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# Loop through all images
for img_key, img_data in annotations.items():
    img_name = img_data['filename']
    img_path = os.path.join(INPUT_DIR, img_name)

    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: couldn't read {img_path}")
        continue

    regions = img_data['regions']
    if isinstance(regions, dict):
        regions = list(regions.values())

    if not regions:
        print(f"No valid regions for {img_name}, skipping.")
        continue

    # Save original image
    original_img_name = f"{os.path.splitext(img_name)[0]}_original.jpg"
    original_img_path = os.path.join(OUTPUT_DIR, original_img_name)
    cv2.imwrite(original_img_path, image)

    # Save original annotation
    augmented_annotations[original_img_name + "0"] = {
        "filename": original_img_name,
        "size": 0,
        "regions": regions
    }

    # Prepare keypoints grouped by region
    polygons = []
    labels = []
    for region in regions:
        shape = region['shape_attributes']
        if shape['name'] != 'polygon':
            continue
        pts = list(zip(shape['all_points_x'], shape['all_points_y']))
        polygons.append(pts)
        labels.append(region['region_attributes'])

    flat_keypoints = [pt for polygon in polygons for pt in polygon]
    split_sizes = [len(polygon) for polygon in polygons]

    for i in range(NUM_AUGMENTATIONS):
        augmented = transform(image=image, keypoints=flat_keypoints)
        aug_image = augmented['image']
        aug_keypoints = augmented['keypoints']

        # Validate keypoint count
        if len(aug_keypoints) != len(flat_keypoints):
            print(f"Keypoint count mismatch for {img_name} augmentation {i}, skipping this one.")
            continue

        # Get dimensions of the augmented image
        h, w = aug_image.shape[:2]

        # Reconstruct polygons
        new_polygons = []
        idx = 0
        for size in split_sizes:
            new_polygons.append(aug_keypoints[idx:idx+size])
            idx += size

        aug_region_data = []
        for poly, label in zip(new_polygons, labels):
            if len(poly) < 3:
                continue
            # Clip keypoints to image boundaries
            x_coords = [max(0, min(int(round(x)), w - 1)) for x, y in poly]
            y_coords = [max(0, min(int(round(y)), h - 1)) for x, y in poly]
            aug_region_data.append({
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": x_coords,
                    "all_points_y": y_coords
                },
                "region_attributes": label
            })

        if not aug_region_data:
            continue

        aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
        output_img_path = os.path.join(OUTPUT_DIR, aug_img_name)
        cv2.imwrite(output_img_path, aug_image)

        augmented_annotations[aug_img_name + "0"] = {
            "filename": aug_img_name,
            "size": 0,
            "regions": aug_region_data
        }

# Save annotations
with open(os.path.join(OUTPUT_DIR, OUTPUT_ANNOTATIONS_FILE), 'w') as f:
    json.dump(augmented_annotations, f, indent=2)

print("Augmentation completed without out-of-bounds keypoints!")
