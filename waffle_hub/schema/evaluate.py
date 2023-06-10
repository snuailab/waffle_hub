from dataclasses import dataclass

from waffle_hub.schema.base_schema import BaseSchema


@dataclass
class ObjectDetectionMetric(BaseSchema):
    mAP: float


@dataclass
class ClassificationMetric(BaseSchema):
    accuracy: float


@dataclass
class InstanceSegmentationMetric(BaseSchema):
    mAP: float


@dataclass
class TextRecognitionMetric(BaseSchema):
    accuracy: float
