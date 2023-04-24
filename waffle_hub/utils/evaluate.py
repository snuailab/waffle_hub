from typing import Union

import torch
from torchmetrics.classification import Accuracy
from torchmetrics.detection import mean_ap

from waffle_hub import TaskType
from waffle_hub.schema.evaluate import (
    ClassificationMetric,
    ObjectDetectionMetric,
)
from waffle_hub.schema.fields import Annotation


def convert_to_torchmetric_format(
    total: list[Annotation], task: TaskType, prediction: bool = False
):

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

        else:
            raise NotImplementedError

    if isinstance(datas[0], dict):
        datas = [
            {k: torch.tensor(v) for k, v in data.items()} for data in datas
        ]
    elif isinstance(datas[0], int):
        datas = torch.tensor(datas)
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


def evaluate_function(
    preds: list[Annotation],
    labels: list[Annotation],
    task: str,
    num_classes: int = None,
    *args,
    **kwargs
) -> Union[ClassificationMetric, ObjectDetectionMetric]:
    preds = convert_to_torchmetric_format(preds, task, prediction=True)
    labels = convert_to_torchmetric_format(labels, task)

    if task == TaskType.CLASSIFICATION:
        return evaluate_classification(
            preds, labels, num_classes, *args, **kwargs
        )
    elif task == TaskType.OBJECT_DETECTION:
        return evaluate_object_detection(
            preds, labels, num_classes, *args, **kwargs
        )
    else:
        raise NotImplementedError
