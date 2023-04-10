# Image

<!-- 
from waffle_utils.log import datetime_now
from waffle_utils.utils import type_validator

from . import BaseField


class Image(BaseField):
    def __init__(
        self,
        # required
        image_id: int,
        file_name: str,
        width: int,
        height: int,
        # optional
        date_captured: str = None,
    ):

        self.image_id = image_id
        self.file_name = file_name
        self.width = width
        self.height = height
        self.date_captured = date_captured

    # properties
    @property
    def image_id(self):
        return self.__image_id

    @image_id.setter
    @type_validator(int)
    def image_id(self, v):
        if v is None:
            raise ValueError("image_id should not be None")
        if v and v < 1:
            raise ValueError("id should be greater than 0.")
        self.__image_id = v

    @property
    def file_name(self):
        return self.__file_name

    @file_name.setter
    @type_validator(str)
    def file_name(self, v):
        self.__file_name = v

    @property
    def width(self):
        return self.__width

    @width.setter
    @type_validator(int)
    def width(self, v):
        self.__width = v

    @property
    def height(self):
        return self.__height

    @height.setter
    @type_validator(int)
    def height(self, v):
        self.__height = v

    @property
    def date_captured(self):
        return self.__date_captured

    @date_captured.setter
    @type_validator(str)
    def date_captured(self, v):
        if v is None:
            self.__date_captured = datetime_now()
        else:
            self.__date_captured = v

    @classmethod
    def new(
        cls,
        image_id: int,
        file_name: str,
        width: int,
        height: int,
        date_captured: str = None,
    ) -> "Image":
        """Image Format

        Args:
            image_id (int): image id. natural number.
            file_name (str): file name. relative file path.
            width (int): image width.
            height (int): image height.
            date_captured (str): date_captured string. "%Y-%m-%d %H:%M:%S"

        Returns:
            Image: image class
        """
        return cls(image_id, file_name, width, height, date_captured)

    def to_dict(self) -> dict:
        """Get Dictionary of Category

        Returns:
            dict: annotation dictionary.
        """

        cat = {
            "image_id": self.image_id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "date_captured": self.date_captured,
        }

        return cat
 -->

Image Format

| Property | Type | Description |
| --- | --- | --- |
| image_id | int | image id. natural number. |
| file_name | str | file name. relative file path. |
| width | int | image width. |
| height | int | image height. |
| date_captured | str | date_captured string. "%Y-%m-%d %H:%M:%S" |

# Category
<!-- 
class Category(BaseField):
    def __init__(
        self,
        # required
        category_id: int,
        supercategory: str,
        name: str,
        # for keypoint detection
        keypoints: list[str] = None,
        skeleton: list[list[int]] = None,
    ):

        self.category_id = category_id
        self.supercategory = supercategory
        self.name = name
        self.keypoints = keypoints
        self.skeleton = skeleton

    # properties
    @property
    def category_id(self):
        return self.__category_id

    @category_id.setter
    @type_validator(int)
    def category_id(self, v):
        if v is None:
            raise ValueError("category_id should not be None")
        if v and v < 1:
            raise ValueError("id should be greater than 0.")
        self.__category_id = v

    @property
    def supercategory(self):
        return self.__supercategory

    @supercategory.setter
    @type_validator(str)
    def supercategory(self, v):
        self.__supercategory = v

    @property
    def name(self):
        return self.__name

    @name.setter
    @type_validator(str)
    def name(self, v):
        self.__name = v

    @property
    def keypoints(self):
        return self.__keypoints

    @keypoints.setter
    @type_validator(list)
    def keypoints(self, v):
        self.__keypoints = v

    @property
    def skeleton(self):
        return self.__skeleton

    @skeleton.setter
    @type_validator(list)
    def skeleton(self, v):
        self.__skeleton = v -->

Category Format

| Property | Type | Description |
| --- | --- | --- |
| category_id | int | category id. natural number. |
| supercategory | str | supercategory name. |
| name | str | category name. |
| keypoints | list[str] | keypoint list. |
| skeleton | list[list[int]] | skeleton list. |

# Annotation
<!-- 

class Annotation(BaseField):
    def __init__(
        self,
        # required
        annotation_id: int,
        image_id: int,
        # optional
        category_id: int = None,
        bbox: list[float] = None,
        segmentation: list[float] = None,
        area: float = None,
        keypoints: list[float] = None,
        num_keypoints: int = None,
        caption: str = None,
        value: float = None,
        #
        iscrowd: int = None,
        score: Union[float, list[float]] = None
    ):

        self.annotation_id = annotation_id
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = bbox
        self.segmentation = segmentation
        self.area = area
        self.keypoints = keypoints
        self.num_keypoints = num_keypoints
        self.caption = caption
        self.value = value
        self.iscrowd = iscrowd
        self.score = score

    # properties
    @property
    def annotation_id(self):
        return self.__annotation_id

    @annotation_id.setter
    @type_validator(int)
    def annotation_id(self, v):
        if v is None:
            raise ValueError("annotation_id should not be None")
        if v and v < 1:
            raise ValueError("id should be greater than 0.")
        self.__annotation_id = v

    @property
    def image_id(self):
        return self.__image_id

    @image_id.setter
    @type_validator(int)
    def image_id(self, v):
        if v is None:
            raise ValueError("image_id should not be None")
        if v and v < 1:
            raise ValueError("id should be greater than 0.")
        self.__image_id = v

    @property
    def category_id(self):
        return self.__category_id

    @category_id.setter
    @type_validator(int)
    def category_id(self, v):
        if v and v < 1:
            raise ValueError("id should be greater than 0.")
        self.__category_id = v

    @property
    def bbox(self):
        return self.__bbox

    @bbox.setter
    @type_validator(list)
    def bbox(self, v):
        if v and len(v) != 4:
            raise ValueError("the length of bbox should be 4.")
        self.__bbox = v

    @property
    def segmentation(self):
        return self.__segmentation

    @segmentation.setter
    @type_validator(list)
    def segmentation(self, v):
        if v and len(v) % 2 != 0 and len(v) < 6:
            raise ValueError(
                "the length of segmentation should be at least 6 and divisible by 2."
            )
        self.__segmentation = v

    @property
    def area(self):
        return self.__area

    @area.setter
    @type_validator(float, strict=False)
    def area(self, v):
        self.__area = v

    @property
    def keypoints(self):
        return self.__keypoints

    @keypoints.setter
    @type_validator(list)
    def keypoints(self, v):
        if v and len(v) % 3 != 0 and len(v) < 2:
            raise ValueError(
                "the length of keypoints should be at least 2 and divisible by 3."
            )
        self.__keypoints = v

    @property
    def num_keypoints(self):
        return self.__num_keypoints

    @num_keypoints.setter
    @type_validator(int)
    def num_keypoints(self, v):
        self.__num_keypoints = v

    @property
    def caption(self):
        return self.__caption

    @caption.setter
    @type_validator(str)
    def caption(self, v):
        self.__caption = v

    @property
    def value(self):
        return self.__value

    @value.setter
    @type_validator(float)
    def value(self, v):
        self.__value = v

    @property
    def iscrowd(self):
        return self.__iscrowd

    @iscrowd.setter
    @type_validator(int)
    def iscrowd(self, v):
        self.__iscrowd = v

    @property
    def score(self):
        return self.__score

    @score.setter
    # @type_validator(float)  # TODO: need to upgrade type_validator
    def score(self, v):
        self.__score = v
 -->

Annotation Format

| Property | Type | Description |
| --- | --- | --- |
| annotation_id | int | annotation id. natural number. |
| image_id | int | image id. natural number. |
| category_id | int | category id. natural number. |
| bbox | list[float] | bounding box. |
| segmentation | list[float] | segmentation. |
| area | float | area. |
| keypoints | list[float] | keypoints. |
| num_keypoints | int | number of keypoints. |
| caption | str | caption. |
| value | float | value. |
| iscrowd | int | iscrowd. |
| score | float | score. |
