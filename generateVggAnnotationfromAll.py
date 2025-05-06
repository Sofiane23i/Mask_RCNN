import os
import json

# Paths
ANNOTATIONS_FILE = './datasets/coco/vgg_annotations_all.json'  # The original full annotation file
TRAIN_DIR = './datasets/coco/train/'  # Folder with training images
VAL_DIR = './datasets/coco/val/'      # Folder with validation images
OUTPUT_TRAIN_ANNOTATIONS = './datasets/coco/train/vgg_annotations.json'
OUTPUT_VAL_ANNOTATIONS = './datasets/coco/val/vgg_annotations.json'

# Load full annotations
with open(ANNOTATIONS_FILE) as f:
    full_annotations = json.load(f)

# Get image filenames in each folder (only image files, normalized lowercase)
train_images = {f.lower() for f in os.listdir(TRAIN_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
val_images = {f.lower() for f in os.listdir(VAL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

# Prepare new annotation dicts
train_annotations = {}
val_annotations = {}

print(f"Found {len(train_images)} training images.")
print(f"Found {len(val_images)} validation images.")
print(f"Found {len(full_annotations)} total annotations.")

# Loop through all entries in full annotations
for key, value in full_annotations.items():
    filename = value['filename'].lower()  # Normalize to lowercase
    if filename in train_images:
        train_annotations[key] = value
    elif filename in val_images:
        val_annotations[key] = value

print(f"Collected {len(train_annotations)} train annotations.")
print(f"Collected {len(val_annotations)} val annotations.")

# Save the new annotation files
with open(OUTPUT_TRAIN_ANNOTATIONS, 'w') as f:
    json.dump(train_annotations, f, indent=2)

with open(OUTPUT_VAL_ANNOTATIONS, 'w') as f:
    json.dump(val_annotations, f, indent=2)

print(f"Train annotations saved to {OUTPUT_TRAIN_ANNOTATIONS}")
print(f"Validation annotations saved to {OUTPUT_VAL_ANNOTATIONS}")
