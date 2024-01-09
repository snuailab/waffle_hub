from waffle_utils.enum import StrEnum as BaseType


class TaskType(BaseType):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"
    TEXT_RECOGNITION = "text_recognition"
    REGRESSION = "regression"

    AGNOSTIC = "agnostic"
