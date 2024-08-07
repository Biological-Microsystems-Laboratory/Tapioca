import torch
import os
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM:
    WEIGHTS_PATH = "F:\Code\SAM-Microfluidic\Weights\sam2_hiera_t.yaml"
    CHECKPOINT_PATH = "F:\Code\SAM-Microfluidic\Weights\sam2_hiera_tiny.pt"

    keys = ["area", "bbox", "predicted_iou", "point_coords", "segmentation"]

    def __init__(self, DEVICE, HOME = os.getcwd()):
        os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.HOME = HOME
        sam2 = build_sam2(self.WEIGHTS_PATH, self.CHECKPOINT_PATH, device ='cuda', apply_postprocessing=False)
        print(f"HOME: {HOME}")
       
        print("WEIGHT:", self.WEIGHTS_PATH)
        print(self.CHECKPOINT_PATH, "; exist:", os.path.isfile(self.CHECKPOINT_PATH))
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)
        return

    def generate(self, img):
        return self.mask_generator.generate(img)
