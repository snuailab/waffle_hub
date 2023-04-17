from typing import Union

import cv2
import numpy as np

from waffle_hub import TaskType
from waffle_hub.schema.fields import Annotation

FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 1.0
FONT_WEIGHT = 2

THICKNESS = 2

# random colors with 1000 categories
colors = np.random.randint(0, 255, (1000, 3), dtype="uint8").tolist()


def draw_classification(
    image: np.ndarray,
    annotation: Annotation,
    names: list[str],
    loc_x: int = 10,
    loc_y: int = 30,
):
    category_id: int = annotation.category_id
    score: float = annotation.score

    image = cv2.putText(
        image,
        f"{names[category_id-1]}" + (f": {score:.2f}" if score else ""),
        (loc_x, loc_y),
        FONT_FACE,
        FONT_SCALE,
        colors[category_id-1],
        FONT_WEIGHT,
    )
    return image


def draw_object_detection(
    image: np.ndarray,
    annotation: Annotation,
    names: list[str],
    score: float = None,
):
    x1, y1, w, h = annotation.bbox
    x2 = x1 + w
    y2 = y1 + h

    category_id: int = annotation.category_id
    score: float = annotation.score
    
    image = cv2.putText(
        image,
        f"{names[category_id-1]}" + (f": {score:.2f}" if score else ""),
        (int(x1), int(y1) - 3),
        FONT_FACE,
        FONT_SCALE,
        colors[category_id-1],
        FONT_WEIGHT,
    )
    image = cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        colors[category_id-1],
        THICKNESS,
    )
    return image


def draw_results(
    image: Union[np.ndarray, str],
    results: list[Annotation],
    names: list[str],
):

    if isinstance(image, str):
        image = cv2.imread(image)

    classification_results = [
        result
        for result in results
        if result.task == TaskType.CLASSIFICATION
    ]
    object_detection_results = [
        result
        for result in results
        if result.task == TaskType.OBJECT_DETECTION
    ]

    for i, result in enumerate(classification_results, start=1):
        image = draw_classification(
            image,
            result,
            names=names,
            loc_x=10,
            loc_y=30 * i,
        )

    for i, result in enumerate(object_detection_results, start=1):
        image = draw_object_detection(image, result, names=names)

    return image
