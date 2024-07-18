import cv2
from PIL import Image # (pip install Pillow)
import numpy as np
import os

# make a black image that is the same size as the original (1080 x 1440)

    image_id = 1
    id = 1
    for file in folder:
        gary_mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        mask = np.array(gary_mask, dtype=bool)
        poly = binary_mask_to_poly(mask)

        bbox = poly.bounding_box

        annotation = {
            'segmentation': np.array(poly.exterior.coords).ravel().tolist(),
            'iscrowd': 0,
            'image_id': folder
            # 'category_id': category_id, (1 = droplet, 2 = obstruction)
            # 'id': annotation_id, iterater for every file in current folder
            'bbox': bbox,
            'area': area
        }
