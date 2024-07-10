import cv2
import math

Gray_image = cv2.imread("mask_1.bmp", cv2.IMREAD_GRAYSCALE)
copy = cv2.cvtColor(Gray_image, cv2.COLOR_GRAY2RGB)
cnt, her = cv2.findContours(Gray_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
Perimeter = cv2.arcLength(cnt[0], True)
Area = cv2.contourArea(cnt[0])
test = cv2.drawContours(copy, cnt, -1, (255,255,0), -1)
Circularity = math.pow(Perimeter, 2) / (4 * math.pi * Area)
cv2.drawContours(Gray_image, cnt, -1, (0, 255, 0), 3)

cv2.imshow('Contours', test)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(round(Circularity, 2))

