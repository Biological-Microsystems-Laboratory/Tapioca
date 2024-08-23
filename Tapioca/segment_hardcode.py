import importlib
import time
from typing import Tuple, Optional
from pathlib import Path


import distinctipy
import numpy as np
import pandas as pd
import torch
import cv2

from Tapioca.Circularity import process_image
from Tapioca.help_me import binary_mask_to_rle_np, draw_mask, expand_bbox, label_droplets_indices, save_mask


class image_segmenter():
    MODEL = "mSAM"
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DEBUG = False
    SAVE_RESULTS = False
    SCALE = False
    COCO = False
    SCALE = 6.0755
    def __init__(self,
                 WEIGHTS: Path,
                 RESULTS: Path = Path(f"Results ({MODEL})"),
                 MODEL: str = "mSAM",
                 DEBUG: bool = False,
                 SAVE_RESULTS: bool = False,
                 COCO: bool = False,
                 SCALE: float = 6.0755
                 ) -> None:
        self.RESULTS_FOLDER = RESULTS
        self.MODEL = MODEL
        self.WEIGHTS = WEIGHTS

        module_name = MODEL.lower()
        module = importlib.import_module(f"Models.{module_name}")
        SAM_class = getattr(module, self.MODEL, None)
        self.SAM_OBJ = SAM_class(self.DEVICE, self.WEIGHTS)

        return

    def check_bound_iterator(self, res, image_dim=(1440, 1080)):
        for dictionary in res:
            if (0 < dictionary["bbox"][0] < image_dim[1] - 1 - dictionary["bbox"][2] and
                    0 < dictionary["bbox"][1] < image_dim[0] - 2 - dictionary["bbox"][3]):
                yield {key: dictionary[key] for key in self.keys if key in dictionary}

    def _ingest(self, FILE: Path) -> Tuple[Optional[np.ndarray], Path]:
        """
        Process an image file.

        Args:
            FILE (Path): The path to the image file.

        Returns:
            Tuple[Optional[np.ndarray], Path]: The processed image (if successful) and the image directory path.
        """
        # Load the image
        img = cv2.imread(str(FILE), cv2.IMREAD_UNCHANGED)
        
        # Check if the image is in uint16 format
        print("starting ingest")
        if img.dtype == "uint16":
            print(f"converting {FILE} to uint8 from uint16")
            
            # Normalize the image to the range [0, 255]
            img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            # Convert the image to uint8 format
            img = img.astype('uint8')
            png_file = FILE.with_suffix('.png')
            if not png_file.exists():
                # Save the processed image as a PNG file
                cv2.imwrite(png_file, img)    
        # Convert the image to RGB format
        print("ingest done")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        return img

    def segment_image(self, image_bgr, img_name:Path, sam_result, DEBUG = False):
        print(f"Number of objects: {len(sam_result)}")
        sam_result = sorted(sam_result, key=lambda segment: (segment["point_coords"][0][0], segment["point_coords"][0][1]))

        # Create the folder to hold all the results

        print(f"Results folder: {str(self.RESULTS_FOLDER)}, exists: {self.RESULTS_FOLDER.exists()}")
        self.index = 0
        if self.DEBUG:
            wrong_folder = self.RESULTS_FOLDER / "not_droplet"
            wrong_folder.mkdir(exist_ok=True)
            print(f"Wrong folder: {wrong_folder}, exists: {wrong_folder.exists()}")
            drop_folder = self.RESULTS_FOLDER / "droplet"
            drop_folder.mkdir(exist_ok=True)
            print(f"drop folder: {drop_folder}, exists: {drop_folder.exists()}")

        final_image = image_bgr.copy()
        df = pd.DataFrame()
        colors = distinctipy.get_colors(len(sam_result))
        colors = [[r * 255, g * 255, b * 255] for r, g, b in colors]

        # Process each segment
        for segment in self.check_bound_iterator(sam_result, image_bgr.shape):
            segment["scale (um/px)"] = self.SCALE
            segment["segmentation"] = (segment["segmentation"] * 255).astype(np.uint8)
            segment["index"] = self.index
            segment["Red"] = colors[segment["index"]][0]
            segment["Green"] = colors[segment["index"]][1]
            segment["Blue"] = colors[segment["index"]][2]

            droplet = process_image(segment)
            # Save mask and update final image
            if not droplet:
                if self.DEBUG:
                    final_image = save_mask(segment, final_image, wrong_folder, DEBUG=DEBUG)
            else:
                if self.DEBUG:
                    final_image = save_mask(segment, final_image, drop_folder, DEBUG=DEBUG)
                    if self.COCO:
                        rle = binary_mask_to_rle_np(segment["segmentation"])
                final_image = draw_mask(final_image, segment["segmentation"], fill_value = (segment["Red"],segment["Green"],segment["Blue"]))
                final_image = label_droplets_indices(final_image, segment,
                                                     text_color=distinctipy.get_text_color((segment["Red"],segment["Green"],segment["Blue"])))
            print(f"Finished processing segment {segment['index']}")

            # Update segment data
            del segment["segmentation"]
            segment.update(expand_bbox(segment))

            # Add segment data to DataFrame
            new = pd.DataFrame.from_dict(segment)
            df = pd.concat([df, new], ignore_index=True)
            self.index += 1


        df.set_index('index')

        print(f"Writing base image: {cv2.imwrite(str(self.RESULTS_FOLDER / 'base.jpg'), image_bgr)}")
        print(f"Writing final result: {cv2.imwrite(str(self.RESULTS_FOLDER / 'total_mask.jpg'), final_image)}")
        df.to_excel(str(self.RESULTS_FOLDER / "results.xlsx"), index=False)
        df.to_csv(str(self.RESULTS_FOLDER / "results.csv"), index=False)


        return final_image

        # Draw circles for droplets on final image
        # final_image = label_droplets_circle(final_image, df)


        # Save results


    def gen_seg(self, FILE: Path):
        print(f"on file: {str(FILE)}")

        start_time = time.time()
        pp_img = self._ingest(FILE)
        sam_res = self.SAM_OBJ.generate(pp_img)
        self.keys = self.SAM_OBJ.keys
        fin_img = self.segment_image(pp_img, FILE, sam_res, DEBUG = self.DEBUG)
        print(f"how long it took:    {time.time() - start_time}")
        return fin_img
        
    # def seg_file(self, FILE: Path):
    #     results = self.gen_seg(FILE)
    #     segment_image(pp_img, img_dir, sam_res, sam.keys)
    #     print(f"how long it took:    {time.time() - start_time}")
        

