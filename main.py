import importlib
import os
from pathlib import Path
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from Circularity import process_image


from help_me import distance_from_origin, ensure_directory, draw_mask, check_bound_iterator, expand_bbox, label_droplets_circle, label_droplets_indices

HOME = os.getcwd()

# print(DEVICE)


FOLDER_PATH = "DropletPNGS"
RESULTS = "Results (SAM2)"
images = os.listdir(FOLDER_PATH)
true_start = time.time()



# SCALE is in pixels/um
def segment_image(image_bgr, img_name, sam_result, keys, SCALE=6.0755):
    print(f"Number of objects: {len(sam_result)}")
    sam_result = sorted(sam_result, key=distance_from_origin)

    # Create the folder to hold all the results
    results_folder = os.path.join(HOME, img_name)
    print(f"Results folder: {results_folder}, exists: {os.path.isfile(results_folder)}")
    ensure_directory(results_folder)

    final_image = image_bgr.copy()
    df = pd.DataFrame()

    # Process each segment
    for segment in check_bound_iterator(sam_result, keys, image_bgr.shape):
        segment["scale (um/px)"] = SCALE
        mask = (segment["segmentation"] * 255).astype(np.uint8)
        droplet = process_image(mask, segment)

        # Save mask and update final image
        if not droplet:
            wrong_folder = os.path.join(results_folder, "not_droplet")
            print(f"Wrong folder: {wrong_folder}, exists: {os.path.isfile(wrong_folder)}")
            final_image = save_mask(segment, final_image, wrong_folder)
        else:
            final_image = save_mask(segment, final_image, results_folder)

        print(f"Finished processing segment {segment['index']}")

        # Update segment data
        del segment["segmentation"]
        segment.update(expand_bbox(segment))

        # Add segment data to DataFrame
        if segment:
            new = pd.DataFrame.from_dict(segment)
            df = pd.concat([df, new], ignore_index=True)

    df.set_index('index')

    # Draw circles for droplets on final image
    final_image = label_droplets_circle(final_image, df)
    final_image = label_droplets_indices(final_image, df)
    

    # Save results
    print(f"Writing base image: {cv2.imwrite(os.path.join(results_folder, 'base.jpg'), image_bgr)}")
    print(f"Writing final result: {cv2.imwrite(os.path.join(results_folder, 'total_mask.jpg'), final_image)}")
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
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)
    cropped_img = img[y:y + h, x:x + w]
    cropped_mask = mask[y:y + h, x:x + w]
    total_mask = draw_mask(img, mask)
    comb_crop = total_mask[y:y + h, x:x + w]
    
    # make combined and cropped mask as a toggleable 
    print("write combined image: " + str(
        cv2.imwrite(f"{comb_folder}/{iterator}.jpg", comb_crop)))  # save just the droplet
    print("write cropped image: " + str(
        cv2.imwrite(f"{droplet_folder}/crop_{iterator}.jpg", cropped_img)))  # save just the droplet
    print("write cropped mask: " + str(
        cv2.imwrite(f'{mask_folder}/crop_mask_{iterator}.jpg', cropped_mask)))  # save droplet mask
    return total_mask

def ingest(file, RESULTS=RESULTS, FOLDER_PATH=FOLDER_PATH):
    FILE = os.path.join(FOLDER_PATH, file)
    img_dir = os.path.join(RESULTS, os.path.splitext(os.path.basename(FILE))[0])
    if os.path.isdir(img_dir):
        test = None
        print(f"already annotated {file}, moving on to next folder")
    else:
        annotated = False
        test = cv2.imread(FILE, cv2.IMREAD_UNCHANGED)
        if test.dtype == "uint16":
            print( f"converting {FILE} to uint8 from uint16")
            print(f"max:{test.max()}, min:{test.min()}")
            test = cv2.normalize(test, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            test = test.astype('uint8')
            # cv2.imshow("test", test)
            # cv2.waitKey(0)
            # print(test)
            print(f"max:{test.max()}, min:{test.min()}, shape:{test.shape}, typeL:{test.dtype}")
            pre, ext = os.path.splitext(FILE)
            png_file = pre + ".png"
            
            # cv2.imwrite(png_file, test)
        test = cv2.cvtColor(test,cv2.COLOR_GRAY2RGB)
    return test, img_dir
            


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



class image_segmenter():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __init__(self, MODEL = "SAM", RESULTS: Path = None) -> None:
        self.MODEL = MODEL
        if RESULTS == None:
            self.RESULTS_FOLDER = Path(f"Results ({MODEL})")
        else:
            self.RESULTS_FOLDER = RESULTS
        #import stuff
        
        print(f"Result Directory: {self.RESULTS_FOLDER.mkdir(exist_ok=True)}")
        
        module_name = MODEL.lower()
        module = importlib.import_module(f"Model_OBJ.{module_name}")
        SAM_class = getattr(module, self.MODEL, None)
        self.SAM_OBJ = SAM_class(self.DEVICE, HOME)
        
        # end of initalizing SAM
        #TODO: change results folder based on mode
        
        
        return
    def _ingest(self,FILE: Path) -> Tuple[Optional[np.ndarray], Path]:
        
        img_dir = self.RESULTS_FOLDER / FILE.stem
        # print(img_dir)
        if img_dir.is_dir():
            test = None
            print(f"already annotated {file}, moving on to next folder")
        else:
            annotated = False
            test = cv2.imread(str(FILE), cv2.IMREAD_UNCHANGED)
            if test.dtype == "uint16":
                print( f"converting {FILE} to uint8 from uint16")
                # print(f"max:{test.max()}, min:{test.min()}")
                test = cv2.normalize(test, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                test = test.astype('uint8')
                png_file = FILE.with_suffix('.png')
                if not png_file.exists():
                    cv2.imwrite(png_file, test)
            test = cv2.cvtColor(test,cv2.COLOR_GRAY2RGB)
        return test, img_dir
    
    def gen_seg(self, FILE: Path):
        print(f"on file: {str(FILE)}")

        start_time = time.time()
        pp_img, img_dir = self._ingest(FILE)
        if pp_img is None:
            print(f"how long it took:    {time.time() - start_time}")    
            return
        sam_res = self.SAM_OBJ.generate(pp_img)
        segment_image(pp_img, img_dir, sam_res, sam.keys)
        print(f"how long it took:    {time.time() - start_time}")   
        
    def seg_file(self, FILE: Path):
        results = self.gen_seg(FILE)
        segment_image(pp_img, img_dir, sam_res, sam.keys)
        print(f"how long it took:    {time.time() - start_time}") 
        

# sam2 = SAM2(HOME) # This line is for SAM 2 bc it wasnt working super well

# for i, version in enumerate(sams):
#     sam = version
#     RESULTS = f"Results ({folder_ver[i]})"
# Specify the directory you want to list

directory_path = Path(FOLDER_PATH)  # Replace with your directory path
def min():
    # List all objects in the specified directory and get their full paths
    test = image_segmenter()
    
    for file in directory_path.iterdir():
        if file.is_file() and file.suffix.lower() in {'.tif', '.tiff'}:
            # print(f"on file: {file}")

            start_time = time.time()
            # test._ingest(file)
            # pp_img, img_dir = ingest(file, RESULTS)
            # if pp_img is not None:
            #     sam_res = sam.generate(pp_img)
            #     segment_image(pp_img, img_dir, sam_res, sam.keys)
            # print(f"how long it took:    {time.time() - start_time}")
    # print(f"total time: {time.time() - true_start} for {len(images)} images")
    # print("hello")
    return

from timeit import timeit
print(timeit(min, number=1))