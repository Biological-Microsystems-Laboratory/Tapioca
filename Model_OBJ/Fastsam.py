import os
from fastsam import FastSAM, FastSAMPrompt

class Fast_SAM:

    MODEL_TYPE = "sam.py"
    keys = ["area", "bbox", "predicted_iou", "point_coords", "segmentation"]

    def __init__(self, DEVICE="cpu", HOME = os.getcwd()):
        self.HOME = HOME
        self.DEVICE = DEVICE
        print(f"HOME: {HOME}")
        self.WEIGHTS_PATH = os.path.join(self.HOME, 'Weights')
        self.CHECKPOINT_PATH = os.path.join(self.WEIGHTS_PATH, "FastSAM.pt")
        self.model = FastSAM(self.CHECKPOINT_PATH)
        return
    def generate(self, img_path):
        everything_results = self.model(img_path, device=self.DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, )
        return self.mask_generator.generate(img)
