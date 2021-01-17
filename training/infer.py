import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

register_coco_instances("unified_train", {}, "unified/train.json", "unified/images/train")

cfg = get_cfg()
cfg.DATASETS.TRAIN = ("unified_train",)
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
predictor = DefaultPredictor(cfg)

im = cv2.imread("test2.jpg")
outputs = predictor(im)
print(outputs)
v = Visualizer(im[:, :, ::-1],
               metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
               scale=1.2
)
    
    

v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Model Output", v.get_image()[:, :, ::-1])
cv2.waitKey(0)  
cv2.destroyAllWindows()  
