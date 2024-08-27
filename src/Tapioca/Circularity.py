import cv2
import math
import matplotlib.pyplot as plt
import numpy as np


def get_contour(large_mask):
    contours, hierarchy = cv2.findContours(large_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def process_image(row):
    # matrix already preprocessed for ease
    contour = get_contour(large_mask=row["segmentation"])
    # print(contour)
    if len(contour) > 1:
        print("check if more than 1")
        row["centroid_x"] = len(contour)
        row["centroid_y"] = len(contour)
        row["droplet"] = False
        return False
    else:
        contour = contour[0]
    Perimeter = cv2.arcLength(contour, True)
    row["perimeter(um)"] = Perimeter / row["scale (um/px)"]
    row["area(um)"] = row["area"] / (row["scale (um/px)"] ** 2)
    row["circularity"] = (row["perimeter(um)"] ** 2) / (4 * math.pi * row["area(um)"])
    

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
        row["droplet"] = False

    row["centroid_x"] = cX
    row["centroid_y"] = cY

    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.circle(mask, (cX, cY), 5, 255, -1)
    # cv2.putText(mask, f"circ: {row["circularity"]}", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # # display the image
    # cv2.imshow("Image", mask)
    # cv2.waitKey(0)
    if (row["circularity"] > 1.16) or (row["area"] <= 1000) or (row["circularity"] < 0.5):
        row["droplet"] = False
    else:
        row["droplet"] = True
        
    return row["droplet"]

# test = cv2.imread("30_30_1/Masks/large_0.bmp", cv2.IMREAD_GRAYSCALE)
# test_norm = cv2.imread("30_30_1/Masks/large_0.bmp", cv2.IMREAD_COLOR)
#
# # cv2.imshow("Processed Image", test)
# print(process_image(test))
