import json
import os
from pathlib import Path
from pycocotools import mask
from Model_OBJ.msam import mSAM
import cv2
import numpy as np

def binary_mask_to_rle_np(binary_mask):
    # rle = {"counts": [], "size": list(binary_mask.shape)}

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))


    return  lengths.tolist()



def create_coco_dataset(output_dir: Path, images_dir: Path, annotations=None):
    categories = ["Droplet"]
    base_dir = output_dir
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create categories
    category_id_map = {}
    for i, categories in enumerate(categories):
        category_id_map[categories] = i + 1
        coco_format['categories'].append({
            "id": i + 1,
            "name": categories,
            "supercategory": "none"
        })

    # iterate through images

    image_id = 0
    SAM = mSAM("cpu")
    for file in images_dir.iterdir():
        if file.is_file() and file.suffix.lower() in {'.tif', '.tiff'}:
            if image_id > 3:
                return
            else:

                output_images = images_dir / "images"

                output_images.mkdir(exist_ok=True)


                img = cv2.imread(file)
                cv2.imwrite(image_dir / file.name, img)


                height, width = img.shape[:2]
                coco_format['images'].append({
                    "id": image_id + 1,
                    "file_name": file.name,
                    "width": width,
                    "height": height
                })
                print(file)

    #            DONE WITH IMAGES NOW PROCESSING
                annotation_id = 1
                image_annotation = SAM.generate(img)
                annotations = image_annotation
                for annotation in annotations:
                    coco_format['annotations'].append({
                        "id": annotation_id,
                        "image_id": image_id + 1,
                        "category_id": 1,
                        "segmentation": binary_mask_to_rle_np(annotation["segmentation"]),
                        "bbox": annotation["bbox"],
                        "area": annotation["area"],
                        "iscrowd": 0
                    })
                    annotation_id += 1

                image_id += 1






    # Save the COCO dataset

    output_file = output_dir / "test.json"

    with open(output_dir, 'w') as f:
        json.dump(coco_format, f)


# Example usage

image_dir = Path('DropletPNGS')
output_dir = Path('Results/dataset')
create_coco_dataset(output_dir, image_dir)
