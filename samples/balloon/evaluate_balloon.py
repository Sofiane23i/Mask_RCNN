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

def compute_metrics(dataset, model, config, iou_threshold=0.5):
    APs, precisions, recalls = [], [], []
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        
        molded_images = np.expand_dims(modellib.mold_image(image.astype(np.float32), config), 0)
        results = model.detect([image], verbose=0)
        r = results[0]
        
        AP, precisions_, recalls_, _ =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'],
                             iou_threshold=iou_threshold)
        APs.append(AP)
        if precisions_.size:
            precisions.append(np.mean(precisions_))
            recalls.append(np.mean(recalls_))
    
    mAP = np.mean(APs) if APs else 0
    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0
    return mAP, mean_precision, mean_recall


def evaluate(model, dataset_dir, config):
    # Load train and val sets
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(dataset_dir, "train")
    dataset_train.prepare()

    dataset_val = BalloonDataset()
    dataset_val.load_balloon(dataset_dir, "val")
    dataset_val.prepare()

    print("\nEvaluating on TRAIN set...")
    mAP_train, precision_train, recall_train = compute_metrics(dataset_train, model, config)
    print(f"TRAIN Results:\nmAP: {mAP_train:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}")

    print("\nEvaluating on VAL set...")
    mAP_val, precision_val, recall_val = compute_metrics(dataset_val, model, config)
    print(f"VAL Results:\nmAP: {mAP_val:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}")

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
