import os
from pathlib import Path

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


class SAM:

    MODEL_TYPE = "vit_h"
    keys = ["area", "bbox", "predicted_iou", "point_coords", "segmentation"]

    def __init__(self, DEVICE, CHECKPOINT_PATH: Path):
        self.CHECKPOINT_PATH = CHECKPOINT_PATH
        print(self.CHECKPOINT_PATH, "; exist:", self.CHECKPOINT_PATH.exists())

        sam_obj = sam_model_registry[self.MODEL_TYPE](checkpoint=self.CHECKPOINT_PATH).to(device=DEVICE)
        self.mask_generator = SamAutomaticMaskGenerator(sam_obj)
        return

    def generate(self, img):
        return self.mask_generator.generate(img)
