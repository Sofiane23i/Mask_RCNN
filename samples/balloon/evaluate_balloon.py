import os
import sys
import argparse
import numpy as np
import skimage.io
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from balloon import BalloonConfig, BalloonDataset  # assuming balloon.py has your dataset & config

from collections import defaultdict
from mrcnn import visualize

def compute_metrics(dataset, model, config, iou_threshold=0.5):
    class_stats = defaultdict(lambda: {'APs': [], 'precisions': [], 'recalls': [], 'count': 0})
    total_images = len(dataset.image_ids)

    for idx, image_id in enumerate(dataset.image_ids, 1):
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

        molded_images = np.expand_dims(modellib.mold_image(image.astype(np.float32), config), 0)
        results = model.detect([image], verbose=0)
        r = results[0]

        AP, precisions_, recalls_, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'],
                             iou_threshold=iou_threshold)

        for class_id in np.unique(gt_class_id):
            class_name = dataset.class_names[class_id]
            mask = gt_class_id == class_id
            class_stats[class_name]['count'] += np.sum(mask)
            if AP is not None:
                class_stats[class_name]['APs'].append(AP)
            if precisions_.size:
                class_stats[class_name]['precisions'].append(np.mean(precisions_))
                class_stats[class_name]['recalls'].append(np.mean(recalls_))

        if idx % 100 == 0 or idx == total_images:
            print(f"Processed {idx}/{total_images} images...")

    # Compute aggregate metrics
    overall_APs = []
    overall_precisions = []
    overall_recalls = []

    print("\nPer-Class Performance:")
    for class_name, stats in class_stats.items():
        if stats['APs']:
            ap = np.mean(stats['APs'])
            pr = np.mean(stats['precisions']) if stats['precisions'] else 0
            rc = np.mean(stats['recalls']) if stats['recalls'] else 0
        else:
            ap = pr = rc = 0
        print(f"  {class_name}: Count={stats['count']} | AP={ap:.4f}, Precision={pr:.4f}, Recall={rc:.4f}")
        overall_APs.extend(stats['APs'])
        overall_precisions.extend(stats['precisions'])
        overall_recalls.extend(stats['recalls'])

    mAP = np.mean(overall_APs) if overall_APs else 0
    mean_precision = np.mean(overall_precisions) if overall_precisions else 0
    mean_recall = np.mean(overall_recalls) if overall_recalls else 0

    return mAP, mean_precision, mean_recall, class_stats


def evaluate(model, dataset_dir, config):
    for split in ['train', 'val']:
        print(f"\nEvaluating on {split.upper()} set...")
        dataset = BalloonDataset()
        dataset.load_balloon(dataset_dir, split)
        dataset.prepare()

        class_counts = defaultdict(int)
        for image_id in dataset.image_ids:
            _, _, gt_class_id, _, _ = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
            for cid in gt_class_id:
                class_counts[dataset.class_names[cid]] += 1

        print(f"\n{split.upper()} SET CLASS COUNTS:")
        print(f"Total images: {len(dataset.image_ids)}")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} instances")

        mAP, precision, recall, _ = compute_metrics(dataset, model, config)
        print(f"\n{split.upper()} Results:")
        print(f"Overall mAP: {mAP:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")


class InferenceConfig(BalloonConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Mask R-CNN on the custom balloon dataset.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/balloon/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=os.path.join(ROOT_DIR, "logs"),
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (optional)')
    args = parser.parse_args()

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=args.logs, config=config)
    print(f"Loading weights from {args.weights}")
    model.load_weights(args.weights, by_name=True)

    evaluate(model, args.dataset, config)
