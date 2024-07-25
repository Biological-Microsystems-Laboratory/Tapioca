import os
import cv2
import numpy as np
import colorsys
from shapely.geometry import Polygon

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


def draw_mask(img, mask, fill_value=None):
    if fill_value is None:
        fill_value = np.random.randint(0, high=255, size=3, dtype=int)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(img, mask=colored_mask, fill_value=fill_value)
    image_overlay = masked.filled()
    masked_image = cv2.addWeighted(img, 0.6, image_overlay, 0.4, 0)

    return masked_image


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")


# def generate_distinct_color_bgr(index):
#     golden_ratio_conjugate = 0.618033988749895
#     hue = (index * golden_ratio_conjugate) % 1
#     rgb = colorsys.hsv_to_rgb(hue, 100, 100)
#     return tuple(int(255 * x) for x in reversed(rgb))


# def binary_matrix_to_polygon(matrix):
#     # Find the coordinates of non-zero elements
#     coords = np.column_stack(np.where(matrix == 1))
#
#     # Create a polygon from the coordinates
#     polygon = Polygon(coords)
#
#     # Simplify the polygon to remove unnecessary vertices
#     simplified_polygon = polygon.simplify(tolerance=0.5, preserve_topology=True)
#
#
#     return simplified_polygon


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
