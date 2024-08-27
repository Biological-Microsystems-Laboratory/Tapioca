
import os
import cv2
import distinctipy
from matplotlib import pyplot as plt
import numpy as np
import colorsys
# from shapely.geometry import Polygon

def binary_mask_to_rle_np(binary_mask):
    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))


    return  lengths.tolist()


def label_droplets_indices(image, row, font_scale=0.5, thickness=1, text_color=(255, 255, 255)):

    font = cv2.FONT_HERSHEY_SIMPLEX

    if row["droplet"]:
        # Convert centroid coordinates to integers
        x = int(row["centroid_x"])
        y = int(row["centroid_y"])

        # Convert index to string
        text = str(row["index"])

        # Get the size of the text
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate the position to center the text on the droplet
        text_x = x - text_width // 2
        text_y = y + text_height // 2

        # Write the index on the image
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)

    return image

def expand_bbox(row):
    if 'bbox' in row:
        x, y, w, h = row['bbox']
        row['x'], row['y'], row['w'], row['h'] = x, y, w, h
        print(x, y, w, h)
        del row['bbox']
    return row


def draw_mask(img, mask, fill_value=None):
    if fill_value is None:
        fill_value = np.random.randint(0, high=255, size=3, dtype=int)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(img, mask=colored_mask, fill_value=fill_value)
    image_overlay = masked.filled()
    masked_image = cv2.addWeighted(img, 0.3, image_overlay, 0.7, 0)
    return masked_image


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")


def save_mask(obj, img, folder_name,
              DEBUG: bool = False):  # file_name is defined while we loop so we create the big folder outside
    """" e.g. HOME
               30_30_1_res
                 Masks
                 Combined
                 Droplets
                 total_mask.jpg
                 results.csv
    """

    folders = ["Masks", "droplets"]
    for i, folder in enumerate(folders):
        folders[i] = (folder_name / folder).mkdir(exist_ok=True)
    mask = obj["segmentation"]
    iterator = obj["index"]
    mask_name = os.path.join(folders[0], f"large_{iterator}.bmp")
    print("write large mask: " + str(cv2.imwrite(mask_name, mask)))  # save binary mask of entire image
    x, y, w, h = obj["bbox"]
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)
    cropped_img = img[y:y + h, x:x + w]
    if DEBUG:
        folders.append(os.path.join(folder_name, "Combined"))
        ensure_directory(folders[2])
        # comb_crop = total_mask[y:y + h, x:x + w]
        cropped_mask = mask[y:y + h, x:x + w]
        # print("write combined image: " + str(cv2.imwrite(f"{folders[2]}/{iterator}.jpg", comb_crop)))  # save just the droplet
        print("write cropped mask: " + str(
            cv2.imwrite(f'{folders[0]}/crop_mask_{iterator}.jpg', cropped_mask)))  # save droplet mask

    # make combined and cropped mask as a toggleable
    print("write cropped image: " + str(
        cv2.imwrite(f"{folders[1]}/crop_{iterator}.jpg", cropped_img)))  # save just the droplet
    return


def distance_from_origin(segment):
    x, y = segment["point_coords"][0]
    return np.sqrt(x**2 + y**2)
