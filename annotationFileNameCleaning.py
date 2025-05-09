import json
import os
import shutil
import re

# Input paths
vgg_path = './datasets/cocoOLD/train/vgg_annotations.json'
src_folder = os.path.dirname(vgg_path)
dst_folder = os.path.join(src_folder, 'train_cleaned')

# Create destination folder
os.makedirs(dst_folder, exist_ok=True)

# Load annotations
with open(vgg_path, 'r') as f:
    data = json.load(f)

new_data = {}

for old_filename, annotation in data.items():
    # Step 1: Clean extension suffix like '.jpg0', '.jpeg0'
    cleaned_filename = re.sub(r'(\.jpe?g)(\d+)$', r'\1', old_filename, flags=re.IGNORECASE)

    # Step 2: Extract base name before '_jpg' and append '.jpg'
    if '_jpg' in cleaned_filename:
        new_name = cleaned_filename.split('_jpg')[0] + '.jpg'
    else:
        new_name = cleaned_filename

    old_image_path = os.path.join(src_folder, cleaned_filename)
    new_image_path = os.path.join(dst_folder, new_name)

    # Check if cleaned file exists before copying
    if os.path.exists(old_image_path):
        shutil.copy2(old_image_path, new_image_path)
        print(f"Copied: {cleaned_filename} -> {new_name}")
    else:
        print(f"[WARNING] Image not found: {old_image_path} — skipping.")
        continue

    # Update annotation
    annotation['filename'] = new_name
    new_data[new_name] = annotation

# Save cleaned annotations
output_json = os.path.join(dst_folder, 'vgg_annotations_cleaned.json')
with open(output_json, 'w') as f:
    json.dump(new_data, f, indent=4)

print(f"\n✅ Done! Cleaned images and annotations saved in: {dst_folder}")
