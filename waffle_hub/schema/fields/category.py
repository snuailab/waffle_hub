from typing import Union

from waffle_utils.utils import type_validator

from waffle_hub import TaskType

from .base_field import BaseField


class Category(BaseField):
    def __init__(
        self,
        # required
        category_id: int,
        name: str,
        supercategory: str = None,
        # for keypoint detection
        keypoints: list[str] = None,
        skeleton: list[list[int]] = None,
        #
        task: Union[str, TaskType] = None,
    ):

        self.category_id = category_id
        self.supercategory = supercategory
        self.name = name
        self.keypoints = keypoints
        self.skeleton = skeleton

        self.task = task

    # properties
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
    def supercategory(self):
        return self.__supercategory

    @supercategory.setter
    def supercategory(self, v):
        self.__supercategory = v if v else "object"

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
        self.__skeleton = v

    @property
    def task(self):
        return self.__task

    @task.setter
    def task(self, v):
        if v is not None and v not in TaskType:
            raise ValueError(f"Invalid task type: {v}" f"Available task types: {list(TaskType)}")
        self.__task = str(v).upper()

    # factories
    @classmethod
    def new(
        cls,
        category_id: int,
        name: str,
        supercategory: str = None,
        keypoints: list[str] = None,
        skeleton: list[list[int]] = None,
        task: Union[str, TaskType] = None,
        **kwargs,
    ) -> "Category":
        """Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            supercategory (str): supercategory name.
            keypoints (list[str]): category name.
            skeleton (list[list[int]]): skeleton edges.
            task (Union[str, TaskType], optional): task type. Default to None.

        Returns:
            Category: category class
        """
        return cls(
            category_id=category_id,
            name=name,
            supercategory=supercategory,
            keypoints=keypoints,
            skeleton=skeleton,
            task=task,
        )

    @classmethod
    def classification(
        cls, category_id: int, name: str, supercategory: str = None, **kwargs
    ) -> "Category":
        """Classification Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            supercategory (str): supercategory name.

        Returns:
            Category: category class
        """
        return cls(
            category_id=category_id,
            name=name,
            supercategory=supercategory,
            task=TaskType.CLASSIFICATION,
        )

    @classmethod
    def object_detection(
        cls, category_id: int, name: str, supercategory: str = None, **kwargs
    ) -> "Category":
        """Object Detection Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            supercategory (str): supercategory name.

        Returns:
            Category: category class
        """
        return cls(
            category_id=category_id,
            name=name,
            supercategory=supercategory,
            task=TaskType.OBJECT_DETECTION,
        )

    @classmethod
    def semantic_segmentation(
        cls, category_id: int, name: str, supercategory: str = None, **kwargs
    ) -> "Category":
        """Segmentation Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            supercategory (str): supercategory name.

        Returns:
            Category: category class
        """
        return cls(
            category_id=category_id,
            name=name,
            supercategory=supercategory,
            task=TaskType.SEMANTIC_SEGMENTATION,
        )

    @classmethod
    def instance_segmentation(
        cls, category_id: int, name: str, supercategory: str = None, **kwargs
    ) -> "Category":
        """Instance Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            supercategory (str): supercategory name.

        Returns:
            Category: category class
        """
        return cls(
            category_id=category_id,
            name=name,
            supercategory=supercategory,
            task=TaskType.INSTANCE_SEGMENTATION,
        )

    @classmethod
    def keypoint_detection(
        cls,
        category_id: int,
        name: str,
        keypoints: list[str],
        skeleton: list[list[int]],
        supercategory: str = None,
        **kwargs,
    ) -> "Category":
        """Keypoint Detection Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            keypoints (list[str]): category name.
            skeleton (list[list[int]]): skeleton edges.
            supercategory (str): supercategory name.

        Returns:
            Category: category class
        """
        return cls(
            category_id=category_id,
            name=name,
            supercategory=supercategory,
            keypoints=keypoints,
            skeleton=skeleton,
            task=TaskType.KEYPOINT_DETECTION,
        )

    @classmethod
    def text_recognition(
        cls, category_id: int, name: str, supercategory: str = None, **kwargs
    ) -> "Category":
        """Text Recognition Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            supercategory (str): supercategory name.

        Returns:
            Category: category class
        """
        return cls(
            category_id=category_id,
            name=name,
            supercategory=supercategory,
            task=TaskType.TEXT_RECOGNITION,
        )

    def to_dict(self) -> dict:
        """Get Dictionary of Category

        Returns:
            dict: annotation dictionary.
        """

        cat = {
            "category_id": self.category_id,
            "supercategory": self.supercategory,
            "name": self.name,
        }

        if self.keypoints is not None:
            cat["keypoints"] = self.keypoints
        if self.skeleton is not None:
            cat["skeleton"] = self.skeleton

        return cat
