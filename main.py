import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

from Circularity import process_image
from Model_OBJ.MOBILE_sam import Mobile_SAM
from help_me import ensure_directory, draw_mask, check_bound_iterator, expand_bbox

HOME = os.getcwd()


# SCALE is in pixels/um
def segment_image(image_bgr, img_name, sam_result, keys, SCALE=6.0755):
    # mask_generator = SamAutomaticMaskGenerator(sam)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # sam_result = mask_generator.generate(image_rgb)  # the actual segmentation of a picture
    print(len(sam_result))  # how many objects we got

    # create the folder to hold all the results
    results_folder = os.path.join(HOME, img_name)
    print(results_folder, "; exist:", os.path.isfile(results_folder))
    ensure_directory(results_folder)

    final_image = image_bgr.copy()
    df = pd.DataFrame()
    # loop over every result while filtering edge cases and saving the images
    for segment in check_bound_iterator(sam_result, keys, image_bgr.shape):
        segment["scale (um/px)"] = SCALE
        # print(segment["index"])
        mask = (segment["segmentation"] * 255).astype(np.uint8)
        not_droplet = process_image(mask, segment)

        if not_droplet:
            wrong_folder = os.path.join(results_folder, "not_droplet")
            print(wrong_folder, "; exist:", os.path.isfile(wrong_folder))
            final_image = save_mask(segment, final_image, wrong_folder)
        else:
            final_image = save_mask(segment, final_image, results_folder)
        print(f"finished {segment['index']}")

        del segment["segmentation"]

        # print(segment.keys())
        segment.update(expand_bbox(segment))
        if segment != {}:
            new = pd.DataFrame.from_dict(segment)
            df = pd.concat([df, new], ignore_index=True)
    df.set_index('index')
    for index, row in df.iterrows():
        if row["droplet"]:
            color = (np.random.randint(0, high=255, size=3, dtype=int)).tolist()
            cv2.circle(final_image, (row["centroid_x"], row["centroid_y"]), 10, color, -1)
    print("writing last results: " + str(cv2.imwrite(os.path.join(results_folder, "total_mask.jpg"), final_image)))
    df.to_excel(os.path.join(results_folder, "results.xlsx"), index=False)
    df.to_csv(os.path.join(results_folder, "results.csv"), index=False)


"""### Output format

`SamAutomaticMaskGenerator` returns a `list` of masks, where each mask is a `dict` containing various information about the mask:

* `segmentation` - `[np.ndarray]` - the mask with `(W, H)` shape, and `bool` type
* `area` - `[int]` - the area of the mask in pixels
* `bbox` - `[List[int]]` - the boundary box of the mask in `xywh` format
* `predicted_iou` - `[float]` - the model's own prediction for the quality of the mask
* `point_coords` - `[List[List[float]]]` - the sampled input point that generated this mask
* `stability_score` - `[float]` - an additional measure of mask quality
* `crop_box` - `List[int]` - the crop of the image used to generate this mask in `xywh` format
"""


def save_mask(obj, img, folder_name):  # file_name is defined while we loop so we create the big folder outside
    """" e.g. HOME
               30_30_1_res
                 Masks
                 Combined
                 Droplets
                 total_mask.jpg
                 results.csv
    """

    # ensure directories
    mask_folder = os.path.join(folder_name, "Masks")
    comb_folder = os.path.join(folder_name, "Combined")
    droplet_folder = os.path.join(folder_name, "droplets")

    ensure_directory(mask_folder)
    ensure_directory(comb_folder)
    ensure_directory(droplet_folder)
    mask = (obj["segmentation"] * 255).astype(np.uint8)
    # area = obj["area"]
    # print(get_contour(mask))

    iterator = obj["index"]
    print("write large mask: " + str(
        cv2.imwrite(f'{mask_folder}/large_{iterator}.bmp', mask)))  # save binary mask of entire image
    x, y, w, h = obj["bbox"]
    cropped_img = img[y:y + h, x:x + w]
    cropped_mask = mask[y:y + h, x:x + w]
    total_mask = draw_mask(img, mask)
    comb_crop = total_mask[y:y + h, x:x + w]
    print("write combined image: " + str(
        cv2.imwrite(f"{comb_folder}/{iterator}.jpg", comb_crop)))  # save just the droplet
    print("write cropped image: " + str(
        cv2.imwrite(f"{droplet_folder}/crop_{iterator}.jpg", cropped_img)))  # save just the droplet
    print("write cropped mask: " + str(
        cv2.imwrite(f'{mask_folder}/crop_mask_{iterator}.jpg', cropped_mask)))  # save droplet mask
    return total_mask


#
# def save_results(results, keys, filename="results.csv"):
#     with open(filename, "w", newline='') as csv_file:
#         # Replace 'bbox' with 'x', 'y', 'w', 'h' in the fieldnames
#         fieldnames = ['index'] + [field if field != 'bbox' else 'x,y,w,h'.split(',') for field in keys]
#         fieldnames = [item for sublist in fieldnames for item in (sublist if isinstance(sublist, list) else [sublist])]
#
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         writer.writeheader()
#
#         for i, res in enumerate(results):
#             # Add the index and expand bbox
#             res_expanded = {'index': i}
#             res_expanded.update(expand_bbox(res))
#
#             writer.writerow(res_expanded)
#
#
# import csv
#
# write_results_to_csv(new_results, keys)
#
# """To Save the Masks"""
#
# def crop_image(img, x, y, w, h):
#     # Crop the image
#     cropped_img = img[y:y + h, x:x + w]
#     # Return the cropped image
#     return cropped_img
#


# TODO: have to include a function to normalize the picture beforehand
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam = Mobile_SAM(DEVICE, HOME)
FOLDER_PATH = "DropletPNGS"
images = os.listdir(FOLDER_PATH)

for file in images:
    print(f"on file: {file}")

    start_time = time.time()

    image_file = os.path.join(FOLDER_PATH, file)

    img_dir = os.path.splitext(os.path.basename(image_file))[0]
    if os.path.isdir(img_dir):
        print(f"already annotated {file}, moving on to next folder")
    else:

        test = cv2.imread(image_file, cv2.IMREAD_COLOR)

        sam_res = sam.generate(test)

        segment_image(test, img_dir, sam_res, sam.keys)
    print(f"how long it took:    {time.time() - start_time}")

print(f"total time: {time.time() - start_time} for {len(images)} images")
# print("hello")
