import json
import os

# Load the CarDDVal format
with open("./datasets/coco/train/vgg_annotations.json", "r") as f:
    new_annotations = json.load(f)

# VGG format output
vgg_format = {}

for image_id, image_data in new_annotations.items():
    filename = image_data.get("filename", image_id)
    regions = []

    for region in image_data.get("regions", []):
        shape_attr = region.get("shape_attributes", {})
        region_attr = region.get("region_attributes", {})

        # Ensure it's a polygon
        if shape_attr.get("name") != "polygon":
            continue

        all_x = shape_attr.get("all_points_x", [])
        all_y = shape_attr.get("all_points_y", [])
        label = region_attr.get("label", "")

        # Build region
        regions.append({
            "shape_attributes": {
                "name": "polygon",
                "all_points_x": all_x,
                "all_points_y": all_y
            },
            "region_attributes": {
                "label": label
            }
        })

    vgg_format[filename] = {
        "filename": filename,
        "size": image_data.get("size", 0),
        "regions": regions,
        "file_attributes": {}
    }

# Save the converted file
with open("converted_to_vgg_format.json", "w") as f:
    json.dump(vgg_format, f, indent=2)

print("✅ Conversion complete. Saved as 'converted_to_vgg_format.json'")