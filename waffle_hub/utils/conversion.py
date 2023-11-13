import cv2
import numpy as np
from pycocotools import mask as mask_utils


def convert_rle_to_mask(rle: dict) -> np.ndarray:
    mask = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])  # height, width
    return mask_utils.decode(mask)


def convert_rle_to_polygon(rle: dict) -> list:
    mask = convert_rle_to_mask(rle)
    return convert_mask_to_polygon(mask)


def convert_mask_to_polygon(mask: np.ndarray) -> list[list]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon = []

    for contour in contours:
        # contour = np.flip(contour, axis=1)
        # polygon.append(contour.ravel().tolist())

        p = []
        contour = contour.reshape(-1, 2)

        if len(contour) < 3:
            continue

        for point in contour:
            p.extend(point.tolist())
        polygon.append(p)
    return polygon


def convert_polygon_to_mask(polygon: list[list], image_size: list[int]) -> np.ndarray:
    """
    Convert polygon to binary mask.

    Parameters:
    - polygon (list[list]): polygon.
    - image_size (list): A list (width, height) specifying the size of the output mask.

    Returns:
    - mask (numpy.ndarray): A binary mask with the polygon filled as white (255) and the background as black (0).
    """

    mask = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    for pts in polygon:
        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def merge_multi_segment(segments: list[list], image_size: tuple) -> list:
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all
    segments into one.
    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    if len(segments) == 0:
        return []
    if len(segments) == 1:
        return segments[0]

    # TODO: improve this function
    # find the coordinates with min distance between each segment
    # then connect these coordinates with one thin line to merge all
    # segments into one

    res = []
    for segment in segments:
        res.extend(segment)
    return res
