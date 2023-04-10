from typing import Union

import cv2
import numpy as np

from waffle_hub.schema.data import ClassificationResult, ObjectDetectionResult

FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 1.0
FONT_WEIGHT = 2

THICKNESS = 2

# random colors with 1000 categories
colors = np.random.randint(0, 255, (1000, 3), dtype="uint8").tolist()


def draw_classification(
    image: np.ndarray,
    category_id: int,
    score: float,
    names: list[str],
    loc_x: int = 10,
    loc_y: int = 30,
):
    image = cv2.putText(
        image,
        f"{names[category_id]}: {score:.2f}",
        (loc_x, loc_y),
        FONT_FACE,
        FONT_SCALE,
        colors[category_id],
        FONT_WEIGHT,
    )
    return image


def draw_object_detection(
    image: np.ndarray,
    category_id: int,
    score: float,
    bbox: list[float],
    names: list[str],
):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    image = cv2.putText(
        image,
        f"{names[category_id]}: {score:.2f}",
        (int(x1), int(y1) - 3),
        FONT_FACE,
        FONT_SCALE,
        colors[category_id],
        FONT_WEIGHT,
    )
    image = cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        colors[category_id],
        THICKNESS,
    )
    return image


def draw_results(
    image: Union[np.ndarray, str],
    results: list[Union[ClassificationResult, ObjectDetectionResult]],
    names: list[str],
):

    if isinstance(image, str):
        image = cv2.imread(image)

    classification_results = [
        result
        for result in results
        if isinstance(result, ClassificationResult)
    ]
    object_detection_results = [
        result
        for result in results
        if isinstance(result, ObjectDetectionResult)
    ]

    for i, result in enumerate(classification_results, start=1):
        image = draw_classification(
            image,
            result.category_id,
            result.score,
            names,
            loc_x=10,
            loc_y=30 * i,
        )

    for i, result in enumerate(object_detection_results, start=1):
        image = draw_object_detection(
            image, result.category_id, result.score, result.bbox, names
        )

    return image
