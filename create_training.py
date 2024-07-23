import cv2
from PIL import Image # (pip install Pillow)
import numpy as np
import os

# get the number of masks

HOME = "" # folder with the same name as the original image
PATH_MASKS = "" # path to masks from the HOME dir
PATH_IMAGES = "" # path to raw images

from glob import glob

print(glob(PATH_MASKS))

