import numpy as np
import cv2


def find_overlap(segment, prev_segment):
    """
    Takes a list of mask file paths and returns the indices of masks that overlap.

    Args:
    mask_paths (list): List of file paths to mask images.

    Returns:
    list: Indices of masks that overlap with any other mask.
    """
    curr_segment = prev_segment + segment["segmentation"]
    if curr_segment.max() > 255:
        # cv2.imshow("overlap", curr_segment)
        # cv2.waitKey(0)
        print("overlap")
        segment["droplet"] = False
        curr_segment = prev_segment
    else:
        print("no overlap")
    return curr_segment