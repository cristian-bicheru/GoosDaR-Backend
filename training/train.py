import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

print("Registering COCO Dataset..")
register_coco_instances("unified_train", {}, "unified/train.json", "unified/images/train")
register_coco_instances("unified_test", {}, "unified/test.json", "unified/images/test")
print("Done!")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("unified_train",)
cfg.DATASETS.TEST = ("unified_test",)
cfg.TEST.EVAL_PERIOD = 1000
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0029999.pth")
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 1e-3
cfg.SOLVER.MAX_ITER = 30000
cfg.SOLVER.NESTEROV = True
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cgf, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = os.path.join("coco_eval")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
