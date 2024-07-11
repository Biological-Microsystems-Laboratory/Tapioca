import os
import cv2
import numpy as np


def select_keys(dictionary, keys, index):
    result = {key: dictionary[key] for key in keys if key in dictionary}
    result['index'] = index
    return result


def check_bound_iterator(res, desired_keys, image_dim=(1440, 1080)):
    for index, dictionary in enumerate(res):
        if (0 < dictionary["bbox"][0] < image_dim[1] - 1 - dictionary["bbox"][2] and
                0 < dictionary["bbox"][1] < image_dim[0] - 2 - dictionary["bbox"][3]):
            yield select_keys(dictionary, desired_keys, index)


def expand_bbox(row):
    if 'bbox' in row:
        x, y, w, h = row['bbox']
        row['x'], row['y'], row['w'], row['h'] = x, y, w, h
        print(x, y, w, h)
        del row['bbox']
    return row


def draw_mask(img, mask):
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(img, mask=colored_mask, fill_value=np.random.randint(0, high=255, size=(3), dtype=int))
    image_overlay = masked.filled()
    masked_image = cv2.addWeighted(img, 0.6, image_overlay, 0.4, 0)

    return masked_image


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

#
# # Test data
# res = [
# {"id": 1, "bbox": [10, 20, 100, 150], "class": "person"},
# {"id": 2, "bbox": [0, 0, 50, 50], "class": "car"},
# {"id": 3, "bbox": [1430, 1070, 20, 20], "class": "dog"},
# {"id": 4, "bbox": [500, 500, 200, 200], "class": "cat"},
# ]
# desired_keys = ["id", "class"]
# image_dim = (1440, 1080)
#
# # Expected output
# expected = [
# {"id": 1, "class": "person"},
# {"id": 4, "class": "cat"}
# ]
#
# # Run the function
# for result in check_bound_iterator(res, desired_keys, image_dim):
#     print(result)
