from mrcnn.convert_coco_to_vgg import convert_coco_to_vgg


convert_coco_to_vgg("./datasets/coco/valid/_annotations.coco.json", "./datasets/coco/valid/vgg_annotations.json", "./")
