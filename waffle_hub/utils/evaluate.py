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
    SemanticSegmentationMetric,
    TextRecognitionMetric,
)
from waffle_hub.schema.fields import Annotation
from waffle_hub.utils.conversion import convert_polygon_to_mask

logger = logging.getLogger(__name__)


def convert_to_torchmetric_format(
    total: list[Annotation], task: TaskType, prediction: bool = False, *args, **kwargs
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

        elif task == TaskType.SEMANTIC_SEGMENTATION:
            data = {
                "boxes": [],
                "masks": [],
                "labels": [],
            }

            for annotation in annotations:
                data["boxes"].append(annotation.bbox)
                data["masks"].append(
                    (
                        convert_polygon_to_mask(
                            annotation.segmentation, image_size=kwargs["image_size"]
                        )
                    ).tolist()
                )
                data["labels"].append(annotation.category_id - 1)

            datas.append(data)

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
    preds: list[Annotation], labels: list[Annotation], num_classes: int, *args, **kwargs
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
    preds: list[Annotation], labels: list[Annotation], num_classes: int, *args, **kwargs
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


def evaluate_instance_segmentation(
    preds: list[Annotation], labels: list[Annotation], num_classes: int, *args, **kwargs
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
    preds: list[Annotation], labels: list[Annotation], num_classes: int, *args, **kwargs
) -> ObjectDetectionMetric:
    preds = convert_to_torchmetric_format(preds, TaskType.TEXT_RECOGNITION, prediction=True)
    labels = convert_to_torchmetric_format(labels, TaskType.TEXT_RECOGNITION)

    correct = reduce(lambda n, pair: n + eq(*pair), zip(preds, labels), 0)
    acc = correct / len(preds)

    result = TextRecognitionMetric(float(acc))
    return result


def evalute_semantic_segmentation(
    preds: list[Annotation],
    labels: list[Annotation],
    num_classes: int,
    image_size: list[int],
    *args,
    **kwargs
) -> SemanticSegmentationMetric:
    preds = convert_to_torchmetric_format(
        preds, TaskType.SEMANTIC_SEGMENTATION, prediction=True, image_size=image_size
    )
    labels = convert_to_torchmetric_format(
        labels, TaskType.SEMANTIC_SEGMENTATION, image_size=image_size
    )
    # TODO: use library

    # mpa
    mean_pixel_accuracy = 0
    for pred, label in zip(preds, labels):
        if pred["masks"].numel() == 0:  # If the object isn't detected
            continue
        
        _mpa = 0
        for label_index, class_id in enumerate(label["labels"]):
            pred_index = torch.where(pred["labels"] == class_id)[0]
            if pred_index.numel() == 0:
                continue

            _mpa += torch.sum(pred["masks"][pred_index] == label["masks"][label_index]) / torch.numel(label["masks"][label_index])
        mean_pixel_accuracy += _mpa / len(label["labels"])
    mean_pixel_accuracy /= len(labels)

    # iou
    iou = 0
    for pred, label in zip(preds, labels):
        if pred["masks"].numel() == 0:  # If the object isn't detected
            continue
        
        _iou = 0
        for label_index, class_id in enumerate(label["labels"]):
            pred_index = torch.where(pred["labels"] == class_id)[0]
            if pred_index.numel() == 0:
                continue

            label_mask = (label["masks"][label_index] == 255)
            pred_mask = (pred["masks"][pred_index] == 255)

            intersection = torch.sum(pred_mask & label_mask)
            union = torch.sum(pred_mask) + torch.sum(label_mask) - intersection

            if union == 0:
                continue

            _iou += (intersection / union)
        iou += _iou / len(label["labels"])
    iou /= len(labels)

    result = SemanticSegmentationMetric(float(mean_pixel_accuracy), float(iou))
    return result


def evaluate_function(
    preds: list[Annotation],
    labels: list[Annotation],
    task: str,
    num_classes: int = None,
    *args,
    **kwargs
) -> Union[
    ClassificationMetric,
    ObjectDetectionMetric,
    InstanceSegmentationMetric,
    TextRecognitionMetric,
    SemanticSegmentationMetric,
]:
    if task == TaskType.CLASSIFICATION:
        return evaluate_classification(preds, labels, num_classes, *args, **kwargs)
    elif task == TaskType.OBJECT_DETECTION:
        return evaluate_object_detection(preds, labels, num_classes, *args, **kwargs)
    elif task == TaskType.INSTANCE_SEGMENTATION:
        return evaluate_instance_segmentation(preds, labels, num_classes, *args, **kwargs)
    elif task == TaskType.TEXT_RECOGNITION:
        return evalute_text_recognition(preds, labels, num_classes, *args, **kwargs)
    elif task == TaskType.SEMANTIC_SEGMENTATION:
        return evalute_semantic_segmentation(preds, labels, num_classes, *args, **kwargs)
    else:
        raise NotImplementedError
