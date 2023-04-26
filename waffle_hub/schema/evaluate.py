from dataclasses import dataclass

from waffle_hub.schema.base_schema import BaseSchema


@dataclass
class ObjectDetectionMetric(BaseSchema):
    mAP: float


@dataclass
class ClassificationMetric(BaseSchema):
    accuracy: float


@dataclass
class SemanticSegmentationMetric(BaseSchema):
    mAP: float
