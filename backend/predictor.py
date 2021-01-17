import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import torch
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import pairwise_ioa, Boxes, Instances

register_coco_instances("unified_train", {}, "unified/train.json", "unified/images/train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "model_V2.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.DATASETS.TRAIN = ("unified_train",)
MetadataCatalog.get("unified_train").thing_classes = ["geese"]
predictor = DefaultPredictor(cfg)

def join_boxes(b1, b2):
    boxes = [b1, b2]
    minX = min([x.tensor[0][0] for x in boxes])
    minY = min([x.tensor[0][1] for x in boxes])
    maxX = max([x.tensor[0][2] for x in boxes])
    maxY = max([x.tensor[0][3] for x in boxes])
    return Boxes(torch.Tensor([minX, minY, maxX, maxY]).reshape(1,4)).to("cuda:0")

def patch(outputs):
    boxes = [Boxes(x.reshape(1,4)) for x in outputs['instances'].pred_boxes]
    if len(boxes) == 0:
        return 0, None
    scores = outputs['instances'].scores

    l = len(boxes)
    i = 0
    j = 0
    while i < l:
        while j < l:
            if i != j:
                if pairwise_ioa(boxes[i], boxes[j]) > 0.9:
                    print()
                    print(boxes[i], boxes[j])
                    boxes[i] = join_boxes(boxes[i], boxes[j])
                    print(boxes[i])
                    boxes.pop(j)
                    scores = torch.cat([scores[:j], scores[j+1:]])
                    l -= 1
                else:
                    j += 1
            else:
                j += 1
        i += 1

    minX = min([x.tensor[0][0] for x in boxes])
    minY = min([x.tensor[0][1] for x in boxes])
    maxX = max([x.tensor[0][2] for x in boxes])
    maxY = max([x.tensor[0][3] for x in boxes])

    outputs['instances'] = Instances(
        outputs['instances'].image_size,
        pred_boxes = Boxes(torch.Tensor([minX, minY, maxX, maxY]).reshape(1,4)),
        scores = torch.Tensor([sum(scores)/len(scores)]),
        pred_classes = torch.tensor([0], dtype=torch.int32)
    )

    return len(boxes), [float(minX), float(minY), float(maxX), float(maxY)]

def infer(sid):
    im = cv2.imread("stills/"+sid+".png")
    outputs = predictor(im)
    num_geese, bbox = patch(outputs)
    print(outputs)

    v = Visualizer(im[:, :, ::-1],
               metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
               scale=1
            )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("stills/"+sid+"_overlay.png", v.get_image()[:, :, ::-1])
    return num_geese, bbox
