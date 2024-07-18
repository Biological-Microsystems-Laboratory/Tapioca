import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


class SAM:

    MODEL_TYPE = "vit_h"
    keys = ["area", "bbox", "predicted_iou", "point_coords", "segmentation"]

    def __init__(self, DEVICE, HOME = os.getcwd()):
        self.HOME = HOME
        print(f"HOME: {HOME}")
        self.WEIGHTS_PATH = os.path.join(self.HOME, 'Weights')
        self.CHECKPOINT_PATH = os.path.join(self.WEIGHTS_PATH, "sam_vit_h_4b8939.pth")

        print("WEIGHT:", self.WEIGHTS_PATH)
        print(self.CHECKPOINT_PATH, "; exist:", os.path.isfile(self.CHECKPOINT_PATH))

        sam_obj = sam_model_registry[self.MODEL_TYPE](checkpoint=self.CHECKPOINT_PATH).to(device=DEVICE)
        self.mask_generator = SamAutomaticMaskGenerator(sam_obj)
        return

    def generate(self, img):
        return self.mask_generator.generate(img)
