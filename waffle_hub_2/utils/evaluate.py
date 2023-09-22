import logging
from functools import reduce
from operator import eq
from typing import Union

import torch
from torchmetrics.classification import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)
from torchmetrics.detection import mean_ap

from waffle_hub import TaskType
from waffle_hub.schema.evaluate import (
    ClassificationMetric,
    InstanceSegmentationMetric,
    ObjectDetectionMetric,
    TextRecognitionMetric,
)
from waffle_hub.schema.fields import Annotation

logger = logging.getLogger(__name__)


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
    preds = convert_to_torchmetric_format(preds, TaskType.CLASSIFICATION, prediction=True)
    labels = convert_to_torchmetric_format(labels, TaskType.CLASSIFICATION)

    mean_acc = Accuracy(task="multiclass", num_classes=num_classes, average="micro")(preds, labels)
    mean_recall = Recall(task="multiclass", num_classes=num_classes, average="micro")(preds, labels)
    mean_precision = Precision(task="multiclass", num_classes=num_classes, average="micro")(
        preds, labels
    )
    mean_f1_score = F1Score(task="multiclass", num_classes=num_classes, average="micro")(
        preds, labels
    )

    accs = Accuracy(task="multiclass", num_classes=num_classes, average="none")(preds, labels)
    recalls = Recall(task="multiclass", num_classes=num_classes, average="none")(preds, labels)
    precisions = Precision(task="multiclass", num_classes=num_classes, average="none")(preds, labels)
    f1_scores = F1Score(task="multiclass", num_classes=num_classes, average="none")(preds, labels)
    confmats = ConfusionMatrix(task="multiclass", num_classes=num_classes)(preds, labels)

    result = ClassificationMetric(
        accuracy=float(mean_acc),
        recall=float(mean_recall),
        precision=float(mean_precision),
        f1_score=float(mean_f1_score),
        accuracy_per_class=accs.tolist(),
        recall_per_class=recalls.tolist(),
        precision_per_class=precisions.tolist(),
        f1_score_per_class=f1_scores.tolist(),
        confusion_matrix=confmats.tolist(),
    )
    return result


def evaluate_object_detection(
    preds: list[Annotation], labels: list[Annotation], num_classes: int
) -> ObjectDetectionMetric:
    preds = convert_to_torchmetric_format(preds, TaskType.OBJECT_DETECTION, prediction=True)
    labels = convert_to_torchmetric_format(labels, TaskType.OBJECT_DETECTION)

    map_dict = mean_ap.MeanAveragePrecision(
        box_format="xywh",
        iou_type="bbox",
        class_metrics=True,
    )(preds, labels)

    result = ObjectDetectionMetric(
        mAP=float(map_dict["map"]),
        mAP_50=float(map_dict["map_50"]),
        mAP_75=float(map_dict["map_75"]),
        mAP_small=float(map_dict["map_small"]),
        mAP_medium=float(map_dict["map_medium"]),
        mAP_large=float(map_dict["map_large"]),
        mAR_1=float(map_dict["mar_1"]),
        mAR_10=float(map_dict["mar_10"]),
        mAR_100=float(map_dict["mar_100"]),
        mAR_small=float(map_dict["mar_small"]),
        mAR_medium=float(map_dict["map_medium"]),
        mAR_large=float(map_dict["map_large"]),
        mAP_per_class=map_dict["map_per_class"].tolist(),
        mAR_100_per_class=map_dict["mar_100_per_class"].tolist(),
    )
    return result


def evaluate_segmentation(
    preds: list[Annotation], labels: list[Annotation], num_classes: int
) -> InstanceSegmentationMetric:
    preds = convert_to_torchmetric_format(preds, TaskType.INSTANCE_SEGMENTATION, prediction=True)
    labels = convert_to_torchmetric_format(labels, TaskType.INSTANCE_SEGMENTATION)

    map_dict = mean_ap.MeanAveragePrecision(
        box_format="xywh",
        iou_type="bbox",
        class_metrics=True,
    )(preds, labels)

    result = InstanceSegmentationMetric(float(map_dict["map"]))
    return result


def evalute_text_recognition(
    preds: list[Annotation], labels: list[Annotation], num_classes: int
) -> ObjectDetectionMetric:
    preds = convert_to_torchmetric_format(preds, TaskType.TEXT_RECOGNITION, prediction=True)
    labels = convert_to_torchmetric_format(labels, TaskType.TEXT_RECOGNITION)

    correct = reduce(lambda n, pair: n + eq(*pair), zip(preds, labels), 0)
    acc = correct / len(preds)

    result = TextRecognitionMetric(float(acc))
    return result


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
