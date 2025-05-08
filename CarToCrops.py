import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# Config
ZOOM_SIZE = 256  # area to zoom into (smaller than original image)
STEP = 128       # how much to move the zoom window
INPUT_IMAGE_DIR = 'val2/'
INPUT_JSON = 'val2/vgg_annotations.json'
OUTPUT_IMAGE_DIR = 'zooms/images/'
OUTPUT_JSON = 'zooms/vgg_annotations.json'

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# Load VGG annotations
with open(INPUT_JSON) as f:
    annotations = json.load(f)

new_annotations = {}
zoom_count = 0

for filename, data in tqdm(annotations.items()):
    image_path = os.path.join(INPUT_IMAGE_DIR, filename)

    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Warning: Could not read image '{image_path}', skipping.")
        continue

    H, W = image.shape[:2]
    regions = data.get('regions', [])

    for y in range(0, H, STEP):
        for x in range(0, W, STEP):
            # Coordinates of the zoom window (before padding)
            x1, y1 = x - ZOOM_SIZE // 2, y - ZOOM_SIZE // 2
            x2, y2 = x1 + ZOOM_SIZE, y1 + ZOOM_SIZE

            # Create padded canvas
            zoom_canvas = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=np.uint8)

            # Valid coordinates inside image
            x1_valid, y1_valid = max(0, x1), max(0, y1)
            x2_valid, y2_valid = min(W, x2), min(H, y2)

            # Paste region from original image to the canvas
            paste_x1 = x1_valid - x1
            paste_y1 = y1_valid - y1
            paste_x2 = paste_x1 + (x2_valid - x1_valid)
            paste_y2 = paste_y1 + (y2_valid - y1_valid)

            zoom_canvas[paste_y1:paste_y2, paste_x1:paste_x2] = image[y1_valid:y2_valid, x1_valid:x2_valid]

            # Resize zoom canvas to full image size
            zoom_resized = cv2.resize(zoom_canvas, (W, H), interpolation=cv2.INTER_LINEAR)

            zoom_filename = f'zoom_{zoom_count}.jpg'
            zoom_path = os.path.join(OUTPUT_IMAGE_DIR, zoom_filename)
            cv2.imwrite(zoom_path, zoom_resized)

            # Adapt annotations
            zoom_regions = []
            for region in regions:
                shape = region.get('shape_attributes', {})
                if shape.get("name") != "polygon":
                    continue

                region_x = np.array(shape.get('all_points_x', []))
                region_y = np.array(shape.get('all_points_y', []))

                if region_x.size == 0 or region_y.size == 0:
                    continue

                # Check if polygon is fully inside zoom window
                if (region_x >= x1_valid).all() and (region_x <= x2_valid).all() and \
                   (region_y >= y1_valid).all() and (region_y <= y2_valid).all():

                    # Shift to zoom window coordinates then scale to full size
                    new_x = ((region_x - x1) / ZOOM_SIZE * W).clip(0, W).astype(int).tolist()
                    new_y = ((region_y - y1) / ZOOM_SIZE * H).clip(0, H).astype(int).tolist()

                    new_shape = {
                        "name": "polygon",
                        "all_points_x": new_x,
                        "all_points_y": new_y
                    }

                    zoom_regions.append({
                        "shape_attributes": new_shape,
                        "region_attributes": region.get('region_attributes', {})
                    })

            if zoom_regions:
                new_annotations[zoom_filename] = {
                    "filename": zoom_filename,
                    "size": os.path.getsize(zoom_path),
                    "regions": zoom_regions,
                    "file_attributes": {}
                }

            zoom_count += 1

# Save updated VGG-style annotation JSON
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(new_annotations, f)

print(f"\n✅ Generated {zoom_count} zoomed images. Annotations saved to {OUTPUT_JSON}")
