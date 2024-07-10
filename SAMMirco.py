
import os
import cv2
import supervision as sv
import csv
import pandas
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt



HOME = os.getcwd()
print("HOME:", HOME)
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

IMAGE_PATH = os.path.join(HOME, "30_30_1.png")

mask_generator = SamAutomaticMaskGenerator(sam)

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

sam_result = mask_generator.generate(image_rgb)

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

def selective_save(res, keys):
  res = sorted(res, key=(lambda x: x['area']), reverse=True)
  filtered_data = [{k: d[k] for k in keys if k in d} for d in res]
  return filtered_data

def check_bound(res, image_dim=(1440, 1080)):
    return [
        dictionary for dictionary in res
        if 0 < dictionary["bbox"][0] < image_dim[0] - 1 - dictionary["bbox"][2] and
           0 < dictionary["bbox"][1] < image_dim[1] - 2 - dictionary["bbox"][3]
    ]

def write_results_to_csv(results, keys,filename="results.csv"):
    def expand_bbox(row):
        if 'bbox' in row:
            x, y, w, h = row['bbox']
            row['x'], row['y'], row['w'], row['h'] = x, y, w, h
            print(x,y,w,h)
            del row['bbox']
        return row


    with open(filename, "w", newline='') as csv_file:
        # Replace 'bbox' with 'x', 'y', 'w', 'h' in the fieldnames
        fieldnames = ['index'] + [field if field != 'bbox' else 'x,y,w,h'.split(',') for field in keys]
        fieldnames = [item for sublist in fieldnames for item in (sublist if isinstance(sublist, list) else [sublist])]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, res in enumerate(results):
            # Add the index and expand bbox
            res_expanded = {'index': i}
            res_expanded.update(expand_bbox(res))

            writer.writerow(res_expanded)

# Example usage:
# new_results = [...]  # Your list of dictionaries
# keys = [...]  # Your list of keys
# write_results_to_csv("results.csv", new_results, keys)

import csv
keys = ["area", "bbox", "predicted_iou", "point_coords"]
print(sam_result[0].keys())
fil_data = selective_save(sam_result, keys)
new_results = check_bound(fil_data)
write_results_to_csv(new_results, keys)

"""To Save the Masks"""

new_results = check_bound(sam_result)
test_case = new_results[0]["segmentation"]

test_case.shape
cv2.imwrite("test.bmp", (test_case*255))

# os.path.join(HOME, "results/masks")

def draw_masks(results, path="results/masks"):
  for i, result in enumerate(results):
    cv2.imwrite(f'{path}/mask_{i}.bmp', result["segmentation"]*255)
  return

draw_masks(new_results)

"""Crop Image to each droplet"""

def crop_image(img, x, y, w, h):
    # Crop the image
    cropped_img = img[y:y+h, x:x+w]
    # Return the cropped image
    return cropped_img

csvFile = pandas.read_csv('results.csv')
row = next(csvFile.iterrows())[1]
test = csvFile[["x","y","w","h"]]
# print(csvFile[["x","y","w","h"]])

i = 0
for index, row in csvFile.iterrows():
  # print(row["x"])
  crop_img = crop_image(image_rgb, row["x"], row['y'], row['w'], row['h'])
  crop_path = os.path.join(os.getcwd(),f"results/result_{i}.png")
  print(crop_path)
  cv2.imwrite(crop_path,crop_img)
  i += 1


# print(len(new_results))

"""### Results visualisation with Supervision

As of version `0.5.0` Supervision has native support for SAM.
"""

mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

detections = sv.Detections.from_sam(sam_result=new_results)

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)
