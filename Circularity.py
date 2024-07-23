import cv2
import math
import matplotlib.pyplot as plt


def get_contour(large_mask):
    contours, hierarchy = cv2.findContours(large_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def process_image(mask, row):
    contour = get_contour(large_mask=mask)
    Perimeter = cv2.arcLength(contour, True)
    row["perimeter(um)"] = Perimeter / row["scale (um/px)"]
    row["area(um)"] = row["area"] / (row["scale (um/px)"] ** 2)
    row["circularity"] = (row["perimeter(um)"] ** 2) / (4 * math.pi * row["area(um)"])

    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    row["centroid_x"] = cX
    row["centroid_y"] = cY

    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.circle(mask, (cX, cY), 5, 255, -1)
    # cv2.putText(mask, f"circ: {row["circularity"]}", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # # display the image
    # cv2.imshow("Image", mask)
    # cv2.waitKey(0)
    if row["circularity"] > 1.16:
        row["droplet"] = False
        return True
    else:
        row["droplet"] = True
        return False

# test = cv2.imread("30_30_1/Masks/large_0.bmp", cv2.IMREAD_GRAYSCALE)
# test_norm = cv2.imread("30_30_1/Masks/large_0.bmp", cv2.IMREAD_COLOR)
#
# # cv2.imshow("Processed Image", test)
# print(process_image(test))
