from dataclasses import dataclass

from waffle_hub.schema.base_schema import BaseSchema


@dataclass
class ObjectDetectionMetric(BaseSchema):
    mAP: float
    mAP_50: float
    mAP_75: float
    mAP_small: float
    mAP_medium: float
    mAP_large: float
    mAR_1: float
    mAR_10: float
    mAR_100: float
    mAR_small: float
    mAR_medium: float
    mAR_large: float
    mAP_per_class: list[float]
    mAR_100_per_class: list[float]


@dataclass
class ClassificationMetric(BaseSchema):
    accuracy: float
    recall: float
    precision: float
    f1_score: float

    accuracy_per_class: list[float]
    recall_per_class: list[float]
    precision_per_class: list[float]
    f1_score_per_class: list[float]

    confusion_matrix: list[list[int]]


@dataclass
class InstanceSegmentationMetric(BaseSchema):
    mAP: float


@dataclass
class TextRecognitionMetric(BaseSchema):
    accuracy: float
