
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import colorsys
# from shapely.geometry import Polygon

def select_keys(dictionary, keys, index):
    result = {key: dictionary[key] for key in keys if key in dictionary}
    result['index'] = index
    return result


def check_bound_iterator(res, desired_keys, image_dim=(1440, 1080)):
    for index, dictionary in enumerate(res):
        if (0 < dictionary["bbox"][0] < image_dim[1] - 1 - dictionary["bbox"][2] and
                0 < dictionary["bbox"][1] < image_dim[0] - 2 - dictionary["bbox"][3]):
            yield select_keys(dictionary, desired_keys, index)

def label_droplets_circle(image, df):
    """
    Draw circles on the image for each droplet in the DataFrame.
    
    Args:
    image (numpy.ndarray): The image to draw on.
    df (pandas.DataFrame): DataFrame containing droplet information.
    
    Returns:
    numpy.ndarray: The image with circles drawn on it.
    """
    for _, row in df.iterrows():
        if row["droplet"]:
            color = np.random.randint(0, high=255, size=3, dtype=int).tolist()
            cv2.circle(image, (int(row["centroid_x"]), int(row["centroid_y"])), 10, color, -1)
    return image


def label_droplets_indices(image, df, font_scale=0.5, thickness=1, text_color=(255, 255, 255)):
    """
    Write the index of each droplet on the image.
    
    Args:
    image (numpy.ndarray): The image to write on.
    df (pandas.DataFrame): DataFrame containing droplet information.
    font_scale (float): Scale of the font. Default is 0.5.
    thickness (int): Thickness of the text. Default is 1.
    text_color (tuple): Color of the text in BGR format. Default is white.
    
    Returns:
    numpy.ndarray: The image with indices written on it.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for index, row in df.iterrows():
        if row["droplet"]:
            # Convert centroid coordinates to integers
            x = int(row["centroid_x"])
            y = int(row["centroid_y"])
            
            # Convert index to string
            text = str(index)
            
            # Get the size of the text
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Calculate the position to center the text on the droplet
            text_x = x - text_width // 2
            text_y = y + text_height // 2
            
            # Draw a small filled rectangle behind the text for better visibility
            cv2.rectangle(image, (text_x - 2, text_y - text_height - 2),
                          (text_x + text_width + 2, text_y + 2),
                          (0, 0, 0), -1)  # Black background
            
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
    masked_image = cv2.addWeighted(img, 0.6, image_overlay, 0.4, 0)

    return masked_image


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

def distance_from_origin(segment):
    x, y = segment["point_coords"][0]
    return np.sqrt(x**2 + y**2)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 


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
