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
    mAP_per_classes: list
    mAR_100_per_class: list


@dataclass
class ClassificationMetric(BaseSchema):
    accuracy: float


@dataclass
class InstanceSegmentationMetric(BaseSchema):
    mAP: float


@dataclass
class TextRecognitionMetric(BaseSchema):
    accuracy: float
