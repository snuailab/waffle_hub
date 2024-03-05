from dataclasses import dataclass

from waffle_hub.schema.base_schema import BaseSchema


@dataclass
class ObjectDetectionMetric(BaseSchema):
    mAP: float
    mAP_50: float
    mAR_100: float
    precision_per_class: list[float]

    f1_score_per_class: list[float] = None
    f1_score: float = None
    
    confusion_matrix: list[dict] = None
    tpfpfn_table: list[float] = None
    
    
    
    mAP_75: float = None
    mAP_small: float = None
    mAP_medium: float = None
    mAP_large: float = None
    mAR_1: float = None
    mAR_10: float = None
    mAR_small: float = None
    mAR_medium: float = None
    mAR_large: float = None
    mAR_100_per_class: list[float] = None
    tpfpfn_table: list[float] = None
    fp_images_set: set = None
    fn_images_set: set = None

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


@dataclass
class SemanticSegmentationMetric(BaseSchema):
    mean_pixel_accuracy: float
    IoU: float
