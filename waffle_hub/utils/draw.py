import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from waffle_utils.file.network import get_file_from_url
from waffle_utils.image.io import load_image

from waffle_hub import TaskType
from waffle_hub.schema.fields import Annotation

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

    # calculate font size and thickness
    font_scale = max(image.shape[0], image.shape[1]) / 1000
    font_scale = 1.0 if font_scale < 1.0 else font_scale
    font_size = int(font_scale * 25)
    thinckness = int(font_scale) * 2

    # download font
    try:
        if not Path(FONT_NAME).exists():
            get_file_from_url(FONT_URL, FONT_NAME, True)

        font = ImageFont.truetype(FONT_NAME, font_size)
    except:
        font = ImageFont.load_default()
        logging.warning("Don't load font file, Using default font.")

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(
        (loc_x, loc_y - font_size),
        f"{names[category_id-1]}" + (f": {score:.2f}" if score else ""),
        font=font,
        fill=tuple(colors[category_id - 1]),
        stroke_width=thinckness,
    )
    image = np.array(img_pil)

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

    # calculate font size and thickness
    font_scale = max(image.shape[0], image.shape[1]) / 1000
    font_scale = 1.0 if font_scale < 1.0 else font_scale
    font_size = int(font_scale * 15)
    thinckness = int(font_scale) * 2

    # download font
    try:
        if not Path(FONT_NAME).exists():
            get_file_from_url(FONT_URL, FONT_NAME, True)
        font = ImageFont.truetype(FONT_NAME, font_size)
    except:
        font = ImageFont.load_default()
        logging.warning("Don't load font file, Using default font.")

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(
        (int(x1), int(y1) - font_size),
        f"{names[category_id-1]}" + (f": {score:.2f}" if score else ""),
        font=font,
        fill=tuple(colors[category_id - 1]),
        stroke_width=thinckness,
    )

    draw.rectangle(
        (int(x1), int(y1), int(x2), int(y2)),
        outline=tuple(colors[category_id - 1]),
        width=thinckness,
    )

    image = np.array(img_pil)

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

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image, "RGBA")
    fill_color = tuple(colors[annotation.category_id - 1])
    fill_color = fill_color + (120,)
    for segment in segments:
        draw.polygon(
            segment,
            fill=fill_color,
        )

    image = np.array(pil_image)

    return image


def draw_text_recognition(
    image: np.ndarray,
    annotation: Annotation,
    loc_x: int = 0,
    loc_y: int = 10,
):
    # calculate font size and thickness
    font_scale = max(image.shape[0], image.shape[1]) / 1000
    font_size = int((0.7 if font_scale < 0.7 else font_scale) * 25)
    thinckness = int(font_scale) * 2

    # download font
    try:
        if not Path(FONT_NAME).exists():
            get_file_from_url(FONT_URL, FONT_NAME, True)
        font = ImageFont.truetype(FONT_NAME, font_size)
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
        stroke_width=thinckness,
    )

    image = np.array(img_pil)

    return image


def draw_results(
    image: Union[np.ndarray, str],
    results: list[Annotation],
    names: list[str],
):

    if isinstance(image, str):
        image = load_image(image)

    task_results = {task: [] for task in TaskType}
    for result in results:
        task_results[result.task.upper()].append(result)

    font_scale = max(image.shape[0], image.shape[1]) / 1000
    font_scale = 1.0 if font_scale < 1.0 else font_scale
    font_size = int(font_scale * 25)
    for i, result in enumerate(task_results[TaskType.CLASSIFICATION], start=1):
        image = draw_classification(
            image,
            result,
            names=names,
            loc_x=10,
            loc_y=font_size * i,
        )

    for i, result in enumerate(task_results[TaskType.OBJECT_DETECTION], start=1):
        image = draw_object_detection(image, result, names=names)

    for i, result in enumerate(task_results[TaskType.INSTANCE_SEGMENTATION], start=1):
        image = draw_instance_segmentation(image, result, names=names)

    for i, result in enumerate(task_results[TaskType.TEXT_RECOGNITION], start=1):
        image = draw_text_recognition(image, result, loc_x=10, loc_y=10)

    return image
