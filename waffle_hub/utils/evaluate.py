from functools import reduce
from operator import eq
from typing import Union

import torch
from torchmetrics.classification import Accuracy
from torchmetrics.detection import mean_ap
from torchmetrics.text import CharErrorRate

from waffle_hub import TaskType
from waffle_hub.schema.evaluate import (
    ClassificationMetric,
    InstanceSegmentationMetric,
    ObjectDetectionMetric,
    TextRecognitionMetric,
)
from waffle_hub.schema.fields import Annotation


def convert_to_torchmetric_format(total: list[Annotation], task: TaskType, prediction: bool = False):

    datas = []
    for annotations in total:

        if task == TaskType.CLASSIFICATION:  # single attribute
            datas.append(annotations[0].category_id - 1)

        elif task == TaskType.OBJECT_DETECTION:
            data = {
                "boxes": [],
                "labels": [],
            }
            if prediction:
                data["scores"] = []

            for annotation in annotations:
                data["boxes"].append(annotation.bbox)
                data["labels"].append(annotation.category_id - 1)
                if prediction:
                    data["scores"].append(annotation.score)

            datas.append(data)

        elif task == TaskType.INSTANCE_SEGMENTATION:
            data = {
                "boxes": [],
                "labels": [],
            }
            if prediction:
                data["scores"] = []

            for annotation in annotations:
                data["boxes"].append(annotation.bbox)
                data["labels"].append(annotation.category_id - 1)
                if prediction:
                    data["scores"].append(annotation.score)

            datas.append(data)

        elif task == TaskType.TEXT_RECOGNITION:
            datas.append(annotations[0].caption)

        else:
            raise NotImplementedError

    if isinstance(datas[0], dict):
        datas = [{k: torch.tensor(v) for k, v in data.items()} for data in datas]
    elif isinstance(datas[0], int):
        datas = torch.tensor(datas)
    elif isinstance(datas[0], str):
        pass
    else:
        raise NotImplementedError

    return datas


def evaluate_classification(
    preds: list[Annotation], labels: list[Annotation], num_classes: int
) -> ClassificationMetric:

    acc = Accuracy(task="multiclass", num_classes=num_classes)(preds, labels)

    return ClassificationMetric(float(acc))


def evaluate_object_detection(
    preds: list[Annotation], labels: list[Annotation], num_classes: int
) -> ObjectDetectionMetric:

    map_dict = mean_ap.MeanAveragePrecision(
        box_format="xywh",
        iou_type="bbox",
        class_metrics=True,
        num_classes=num_classes,
    )(preds, labels)

    return ObjectDetectionMetric(float(map_dict["map"]))


def evaluate_segmentation(
    preds: list[Annotation], labels: list[Annotation], num_classes: int
) -> InstanceSegmentationMetric:

    map_dict = mean_ap.MeanAveragePrecision(
        box_format="xywh",
        iou_type="bbox",
        class_metrics=True,
        num_classes=num_classes,
    )(preds, labels)

    return InstanceSegmentationMetric(float(map_dict["map"]))


def evalute_text_recognition(
    preds: list[Annotation], labels: list[Annotation], num_classes: int
) -> ObjectDetectionMetric:

    correct = reduce(lambda n, pair: n + eq(*pair), zip(preds, labels), 0)
    acc = correct / len(preds)

    return TextRecognitionMetric(float(acc))


def evaluate_function(
    preds: list[Annotation],
    labels: list[Annotation],
    task: str,
    num_classes: int = None,
    *args,
    **kwargs
) -> Union[
    ClassificationMetric, ObjectDetectionMetric, InstanceSegmentationMetric, TextRecognitionMetric
]:
    preds = convert_to_torchmetric_format(preds, task, prediction=True)
    labels = convert_to_torchmetric_format(labels, task)

    if task == TaskType.CLASSIFICATION:
        return evaluate_classification(preds, labels, num_classes, *args, **kwargs)
    elif task == TaskType.OBJECT_DETECTION:
        return evaluate_object_detection(preds, labels, num_classes, *args, **kwargs)
    elif task == TaskType.INSTANCE_SEGMENTATION:
        return evaluate_segmentation(preds, labels, num_classes, *args, **kwargs)
    elif task == TaskType.TEXT_RECOGNITION:
        return evalute_text_recognition(preds, labels, num_classes, *args, **kwargs)
    else:
        raise NotImplementedError
