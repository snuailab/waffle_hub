from abc import ABC, abstractmethod

from waffle_utils.file import io

from waffle_hub import TaskType


class BaseField(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def new(cls):
        pass

    @classmethod
    def from_dict(cls, d: dict, task: str = None) -> "BaseField":

        if task == TaskType.CLASSIFICATION:
            return cls.classification(**d)
        elif task == TaskType.OBJECT_DETECTION:
            return cls.object_detection(**d)
        elif task == TaskType.SEMANTIC_SEGMENTATION:
            return cls.semantic_segmentation(**d)
        elif task == TaskType.INSTANCE_SEGMENTATION:
            return cls.instance_segmentation(**d)
        elif task == TaskType.KEYPOINT_DETECTION:
            return cls.keypoint_detection(**d)
        elif task == TaskType.TEXT_RECOGNITION:
            return cls.text_recognition(**d)
        elif task == TaskType.REGRESSION:
            return cls.regression(**d)
        else:
            return cls.new(**d)

    @classmethod
    def from_json(cls, f: str, task: str = None) -> "BaseField":
        """Load Field from json file.

        Args:
            f (str): json file path.
            task (str, optional): task name. Default to None.

        Returns:
            Field Object: Field Object.
        """
        d: dict = io.load_json(f)
        return cls.from_dict(d, task)
