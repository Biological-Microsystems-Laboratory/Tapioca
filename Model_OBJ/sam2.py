import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import timeit
import matplotlib.pyplot as plt
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    plt.imshow(img)
    
def main():
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    image = Image.open(r'Results\30_30_1\base.jpg')
    image = np.array(image.convert("RGB"))

    import os
    print(os.getcwd())
    checkpoint = "F:\Code\SAM-Microfluidic\Weights\sam2_hiera_tiny.pt"
    model_cfg = "F:\Code\SAM-Microfluidic\Weights\sam2_hiera_t.yaml"

    sam2 = build_sam2(model_cfg, checkpoint, device ='cuda', apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    masks = mask_generator.generate(image)
    show_anns(masks)
    plt.show()
    

main()