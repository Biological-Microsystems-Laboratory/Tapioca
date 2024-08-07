import cv2
from PIL import Image # (pip install Pillow)
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from help_me import ensure_directory

# get the number of masks

HOME = "30_30_1" # folder with the same name as the original image
SEG_HOME = "Results"
PATH_MASKS = "30_30_1/Masks" # path to masks from the HOME dir
PATH_IMAGES = "DropletPNGS" # path to raw images
GT_PATH = "MASKS"

for file in os.listdir(PATH_IMAGES):
    seg_dir = os.path.join(SEG_HOME,os.path.splitext(file)[0])
    ensure_directory(GT_PATH)

    if os.path.exists(seg_dir):
        print(f"path exists {seg_dir}")
        original_image = cv2.imread(os.path.join(PATH_IMAGES,file))
        PATH_MASKS = os.path.join(seg_dir, "Masks")
        all_masks = os.listdir(PATH_MASKS)
        large_masks = list(filter(lambda large: "large_" in large, all_masks))
        print(large_masks)
        CURR_DIR = os.path.basename(seg_dir)
        MASK_RES = os.path.join(GT_PATH, CURR_DIR)
        total_mask = np.zeros((original_image.shape[0],original_image.shape[1]), np.uint16)
        for i, mask in enumerate(large_masks):

            i = i + 1
            print(f"iter: {i}")
            mask_path = os.path.join(PATH_MASKS,mask)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            ret, mask = cv2.threshold(mask_img, 120, 255, cv2.THRESH_BINARY)
            
           
            # plt.imshow(mask)
            # plt.show()
            # cv2.waitKey(0)
            # print(mask_img)
            # print(mask, hue)
            # plt.imshow(total_mask)
            # plt.show()
            total_mask += mask_img
            if np.any(total_mask > 255):
                print("Values bigger than 10 =", total_mask[total_mask > 255])
                plt.imshow(total_mask)
                plt.show()            
        # cv2.imwrite((MASK_RES + ".tif"), total_mask)
        # plt.imshow(total_mask)
        # plt.show()


    else:
        print(f"not annotated: {seg_dir}")


