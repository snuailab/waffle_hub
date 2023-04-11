from dataclasses import dataclass
from typing import Union

from waffle_hub.schema.base_schema import BaseSchema


@dataclass
class ImageInfo(BaseSchema):
    """ImageInfo

    image_path: Image path
    ori_shape: Original image shape (Width, Height)
    new_shape: Resized image shape without padding (Width, Height)
    input_shape: Resized image shape with padding (Width, Height)
    pad: Padding (Left, Top)

    Returns:
        ImageInfo: ImageInfo
    """

    ori_shape: list[int]
    new_shape: list[int]
    input_shape: list[int]
    pad: list[int]
    image_path: str = None


@dataclass
class ClassificationResult(BaseSchema):
    """Classification Result

    category_id: Category ID
    score: Score

    Returns:
        ClassificationResult: ClassificationResult
    """

    category_id: int
    score: float


@dataclass
class ObjectDetectionResult(BaseSchema):
    """Object Detection Result

    category_id: Category ID
    score: Score
    bbox: Box (Left, Top, Width, Height)
    area: Area

    Returns:
        ObjectDetectionResult: ObjectDetectionResult
    """

    category_id: int
    score: float
    bbox: list[float]
    area: float
