import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon, box

# Config
ZOOM_SIZE = 192
STEP = 120
INPUT_IMAGE_DIR = 'datasets/coco/val_cleaned/'
INPUT_JSON = 'datasets/coco/val_cleaned/vgg_annotations_cleaned.json'
OUTPUT_IMAGE_DIR = 'datasets/coco/val/'
OUTPUT_JSON = 'datasets/coco/val/vgg_annotations.json'

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# Load VGG annotations
with open(INPUT_JSON) as f:
    annotations = json.load(f)

new_annotations = {}
zoom_count = 0

for filename, data in tqdm(annotations.items()):
    regions = data.get('regions', [])
    if not regions:
        continue

    image_path = os.path.join(INPUT_IMAGE_DIR, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️ Warning: Could not read image '{image_path}', skipping.")
        continue

    H, W = image.shape[:2]

    for y in range(0, H, STEP):
        for x in range(0, W, STEP):
            x1, y1 = x - ZOOM_SIZE // 2, y - ZOOM_SIZE // 2
            x2, y2 = x1 + ZOOM_SIZE, y1 + ZOOM_SIZE

            zoom_canvas = np.zeros((ZOOM_SIZE, ZOOM_SIZE, 3), dtype=np.uint8)

            x1_valid, y1_valid = max(0, x1), max(0, y1)
            x2_valid, y2_valid = min(W, x2), min(H, y2)

            paste_x1 = x1_valid - x1
            paste_y1 = y1_valid - y1
            paste_x2 = paste_x1 + (x2_valid - x1_valid)
            paste_y2 = paste_y1 + (y2_valid - y1_valid)

            zoom_canvas[paste_y1:paste_y2, paste_x1:paste_x2] = image[y1_valid:y2_valid, x1_valid:x2_valid]

            zoom_resized = cv2.resize(zoom_canvas, (W, H), interpolation=cv2.INTER_LINEAR)

            zoom_regions = []
            for region in regions:
                shape = region.get('shape_attributes', {})
                if shape.get("name") != "polygon":
                    continue

                region_x = np.array(shape.get('all_points_x', []))
                region_y = np.array(shape.get('all_points_y', []))

                if region_x.size == 0 or region_y.size == 0:
                    continue

                polygon = Polygon(zip(region_x, region_y))
                if not polygon.is_valid or polygon.area == 0:
                    continue

                crop_box = box(x1_valid, y1_valid, x2_valid, y2_valid)
                intersection = polygon.intersection(crop_box)

                if intersection.is_empty or not intersection.is_valid:
                    continue

                # Handle MultiPolygon by using the largest one
                if intersection.geom_type == "MultiPolygon":
                    intersection = max(intersection.geoms, key=lambda p: p.area)
                elif intersection.geom_type != "Polygon":
                    continue

                # Keep only if 80% or more of original area is preserved
                if intersection.area / polygon.area < 0.8:
                    continue

                new_x, new_y = zip(*[
                    ((pt[0] - x1) / ZOOM_SIZE * W, (pt[1] - y1) / ZOOM_SIZE * H)
                    for pt in list(intersection.exterior.coords)
                ])

                new_x = np.clip(np.array(new_x).astype(int), 0, W).tolist()
                new_y = np.clip(np.array(new_y).astype(int), 0, H).tolist()

                if len(new_x) < 3 or len(new_y) < 3:
                    continue

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
                zoom_filename = f'zoom_{zoom_count}.jpg'
                zoom_path = os.path.join(OUTPUT_IMAGE_DIR, zoom_filename)
                cv2.imwrite(zoom_path, zoom_resized)

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
    json.dump(new_annotations, f, indent=4)

print(f"\n✅ Generated {zoom_count} zoomed images with annotations. Saved to '{OUTPUT_JSON}'")
