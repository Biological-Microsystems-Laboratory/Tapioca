
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json

#%%

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        with open(f'Ann_{i}.json', 'w') as json_file:
            cutout_and_dump(ann, ["segmentation", "point_coords", "stability_score", "crop_box"], json_file)
        img[m] = color_mask
    ax.imshow(img)


def cutout_and_dump(data, keys_to_remove, json_file):
    # Create a new dictionary excluding the specified keys
    filtered_data = {k: v for k, v in data.items() if k not in keys_to_remove}

    # Dump the filtered dictionary to JSON
    return json.dump(filtered_data, json_file)
#%%
## Real Code
#%%
image = cv2.imread('DropletPNGS/30_30_1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()
#%%
# Generate the mask generator object
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cpu")

mask_generator = SamAutomaticMaskGenerator(sam)

#%%
masks = mask_generator.generate(image)
#%%
# print(masks)
plt.imshow(image)
show_anns(masks)
plt.show()