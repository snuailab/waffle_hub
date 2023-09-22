import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from waffle_utils.file.network import get_file_from_url

from waffle_hub import TaskType
from waffle_hub.schema.fields import Annotation

FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 1.0
FONT_WEIGHT = 2
THICKNESS = 2
FONT_URL = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/fonts/gulim.ttc"
FONT_NAME = "gulim.ttc"


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
        colors[category_id - 1],
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
        colors[category_id - 1],
        FONT_WEIGHT,
    )
    image = cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        colors[category_id - 1],
        THICKNESS,
    )
    return image


def draw_instance_segmentation(
    image: np.ndarray,
    annotation: Annotation,
    names: list[str],
    score: float = None,
):
    image = draw_object_detection(image, annotation, names, score)
    segments: list = annotation.segmentation

    if len(segments) == 0:
        return image

    alpha = np.zeros_like(image)
    for segment in segments:
        segment = np.array(segment).reshape(-1, 2).astype(int)
        alpha = cv2.fillPoly(
            alpha,
            [segment],
            colors[annotation.category_id - 1],
        )
    mask = alpha > 0
    image[mask] = cv2.addWeighted(alpha, 0.3, image, 0.7, 0)[mask]

    return image


def draw_text_recognition(
    image: np.ndarray,
    annotation: Annotation,
    loc_x: int = 0,
    loc_y: int = 10,
):
    # download font
    try:
        global FONT_NAME
        if not Path(FONT_NAME).exists():
            get_file_from_url(FONT_URL, FONT_NAME, True)
        font = ImageFont.truetype(FONT_NAME, int(FONT_SCALE) * 25)
    except:
        font = ImageFont.load_default()
        logging.warning("Don't load font file, Using default font.")

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(
        (loc_x, loc_y),
        annotation.caption,
        font=font,
        fill=tuple(colors[0]),
        stroke_width=FONT_WEIGHT,
    )

    image = np.array(img_pil)

    return image


def draw_results(
    image: Union[np.ndarray, str],
    results: list[Annotation],
    names: list[str],
):

    if isinstance(image, str):
        image = cv2.imread(image)

    task_results = {task: [] for task in TaskType}
    for result in results:
        task_results[result.task.upper()].append(result)

    for i, result in enumerate(task_results[TaskType.CLASSIFICATION], start=1):
        image = draw_classification(
            image,
            result,
            names=names,
            loc_x=10,
            loc_y=30 * i,
        )

    for i, result in enumerate(task_results[TaskType.OBJECT_DETECTION], start=1):
        image = draw_object_detection(image, result, names=names)

    for i, result in enumerate(task_results[TaskType.INSTANCE_SEGMENTATION], start=1):
        image = draw_instance_segmentation(image, result, names=names)

    for i, result in enumerate(task_results[TaskType.TEXT_RECOGNITION], start=1):
        image = draw_text_recognition(image, result, loc_x=10, loc_y=10)

    return image
