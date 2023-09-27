from itertools import permutations

import numpy as np
import pytest

from waffle_hub.schema.evaluate import (
    ClassificationMetric,
    InstanceSegmentationMetric,
    ObjectDetectionMetric,
)
from waffle_hub.schema.fields import Annotation
from waffle_hub.utils.data import resize_image
from waffle_hub.utils.evaluate import (
    evaluate_classification,
    evaluate_object_detection,
    evaluate_segmentation,
)


def test_evaluate_classification():
    result: ClassificationMetric = evaluate_classification(
        preds=[
            [Annotation.classification(category_id=1)],
            [Annotation.classification(category_id=2)],
            [Annotation.classification(category_id=1)],
        ],
        labels=[
            [Annotation.classification(category_id=1)],
            [Annotation.classification(category_id=2)],
            [Annotation.classification(category_id=3)],
        ],
        num_classes=3,
    )

    assert abs(result.accuracy - 2 / 3) < 1e-5


def test_evaluate_object_detection():
    result: ObjectDetectionMetric = evaluate_object_detection(
        preds=[
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=3, bbox=[0, 0, 1, 1], score=1.0),
            ],
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
            ],
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
            ],
        ],
        labels=[
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=3, bbox=[0, 0, 1, 1], score=1.0),
            ],
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=3, bbox=[0, 0, 1, 1], score=1.0),
            ],
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=3, bbox=[0, 0, 1, 1], score=1.0),
            ],
        ],
        num_classes=3,
    )

    assert result.mAP < 1.0


def test_resize_function():
    letter_box = [False, True]
    image_size = [224, 480, 512, 640, 1280, 1600, 1920]

    rect_list = permutations(image_size, 2)
    square_list = [(sz, sz) for sz in image_size]

    for lb in letter_box:
        # square 2 square
        for ori_shape in square_list:
            for resize_shape in square_list:
                image, info = resize_image(
                    np.random.randint(0, 255, ori_shape + (3,), dtype=np.uint8),
                    resize_shape,
                    letter_box=lb,
                )
                assert (image.shape[1], image.shape[0]) == info[
                    "new_shape"
                ], f"Difference from resize image shape to info (letter_box: {lb})"
                if lb:
                    assert (
                        (info["input_shape"][0] - info["new_shape"][0]) // 2,
                        (info["input_shape"][1] - info["new_shape"][1]) // 2,
                    ) == info["pad"], f"Difference Pad info (letter_box: {lb})"
                else:
                    assert (
                        info["input_shape"] == info["new_shape"]
                    ), f"Difference from info[input_shape] to info[new_shape] (letter_box: {lb})"
        # square 2 rect
        for ori_shape in square_list:
            for resize_shape in rect_list:
                image, info = resize_image(
                    np.random.randint(0, 255, ori_shape + (3,), dtype=np.uint8),
                    resize_shape,
                    letter_box=lb,
                )
                assert (image.shape[1], image.shape[0]) == info[
                    "new_shape"
                ], f"Difference from resize image shape to info (letter_box: {lb})"
                if lb:
                    assert (
                        (info["input_shape"][0] - info["new_shape"][0]) // 2,
                        (info["input_shape"][1] - info["new_shape"][1]) // 2,
                    ) == info["pad"], f"Difference Pad info (letter_box: {lb})"
                else:
                    assert (
                        info["input_shape"] == info["new_shape"]
                    ), f"Difference from info[input_shape] to info[new_shape] (letter_box: {lb})"
        # rect 2 square
        for ori_shape in rect_list:
            for resize_shape in square_list:
                image, info = resize_image(
                    np.random.randint(0, 255, ori_shape + (3,), dtype=np.uint8),
                    resize_shape,
                    letter_box=lb,
                )
                assert (image.shape[1], image.shape[0]) == info[
                    "new_shape"
                ], f"Difference from resize image shape to info (letter_box: {lb})"
                if lb:
                    assert (
                        (info["input_shape"][0] - info["new_shape"][0]) // 2,
                        (info["input_shape"][1] - info["new_shape"][1]) // 2,
                    ) == info["pad"], f"Difference Pad info (letter_box: {lb})"
                else:
                    assert (
                        info["input_shape"] == info["new_shape"]
                    ), f"Difference from info[input_shape] to info[new_shape] (letter_box: {lb})"
        # rect 2 rect
        for ori_shape in rect_list:
            for resize_shape in rect_list:
                image, info = resize_image(
                    np.random.randint(0, 255, ori_shape + (3,), dtype=np.uint8),
                    resize_shape,
                    letter_box=lb,
                )
                assert (image.shape[1], image.shape[0]) == info[
                    "new_shape"
                ], f"Difference from resize image shape to info (letter_box: {lb})"
                if lb:
                    assert (
                        (info["input_shape"][0] - info["new_shape"][0]) // 2,
                        (info["input_shape"][1] - info["new_shape"][1]) // 2,
                    ) == info["pad"], f"Difference Pad info (letter_box: {lb})"
                else:
                    assert (
                        info["input_shape"] == info["new_shape"]
                    ), f"Difference from info[input_shape] to info[new_shape] (letter_box: {lb})"
