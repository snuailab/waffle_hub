from dataclasses import dataclass

from waffle_hub.schema.base_schema import BaseSchema
from waffle_utils.log import datetime_now

@dataclass
class DatasetInfo(BaseSchema):
    name: str
    task: str
    created: str = None

    def __post_init__(self):
        self.created = self.created or datetime_now()


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
