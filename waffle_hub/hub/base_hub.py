import logging
from abc import abstractmethod
from functools import cached_property
from pathlib import Path

from waffle_utils.file import io
from waffle_utils.utils import type_validator

logger = logging.getLogger(__name__)


class BaseHub:

    AVAILABLE_TASK = []
    AVAILABLE_MODEL = []
    AVAILABLE_SIZE = []

    # directory settings
    DEFAULT_ROOT_DIR = Path("./models")

    TRAIN_DIR = Path("trains")
    RAW_TRAIN_DIR = Path("raw_train")

    INFERENCE_DIR = Path("inferences")
    EVALUATION_DIR = Path("evaluations")
    EXPORT_DIR = Path("exports")

    # common files
    TRAIN_OPTION_FILE = "configs/common.yaml"
    LAST_CKPT_FILE = "weights/last_ckpt.pth"
    BEST_CKPT_FILE = "weights/best_ckpt.pth"
    METRIC_FILE = "metrics.txt"

    def __init__(
        self,
        name: str,
        task: str = None,
        model_name: str = None,
        model_size: str = None,
        pretrained_model: str = None,
        root_dir: str = None,
    ):

        self.name = name
        self.task = task
        self.model_name = model_name
        self.model_size = model_size
        self.pretrained_model = pretrained_model
        self.root_dir = root_dir

        self.validate_model()

    # properties
    @property
    def name(self):
        return self.__name

    @name.setter
    @type_validator(str)
    def name(self, v):
        self.__name = v

    @property
    def root_dir(self) -> Path:
        return self.__root_dir

    @root_dir.setter
    @type_validator(Path, strict=False)
    def root_dir(self, v):
        self.__root_dir = Path(v) if v else BaseHub.DEFAULT_ROOT_DIR

    @property
    def task(self) -> str:
        return self.__task

    @task.setter
    @type_validator(str)
    def task(self, v):
        if v not in self.AVAILABLE_TASK:
            raise ValueError(
                f"Task {v} is not supported. Choose one of {self.AVAILABLE_TASK}"
            )
        self.__task = v

    @property
    def model_name(self) -> str:
        return self.__model_name

    @model_name.setter
    @type_validator(str)
    def model_name(self, v):
        if v not in self.AVAILABLE_MODEL:
            raise ValueError(
                f"Model {v} is not supported. Choose one of {self.AVAILABLE_MODEL}"
            )
        self.__model_name = v

    @property
    def model_size(self) -> str:
        return self.__model_size

    @model_size.setter
    @type_validator(str)
    def model_size(self, v):
        if v not in self.AVAILABLE_SIZE:
            raise ValueError(
                f"Model Size {v} is not supported. Choose one of {self.AVAILABLE_SIZE}"
            )
        self.__model_size = v

    @property
    def pretrained_model(self) -> str:
        return self.__pretrained_model

    @pretrained_model.setter
    @type_validator(str)
    def pretrained_model(self, v):
        self.__pretrained_model = v

    @cached_property
    def model_dir(self) -> Path:
        return self.root_dir / self.name

    @cached_property
    def train_dir(self) -> Path:
        return self.model_dir / BaseHub.TRAIN_DIR

    @cached_property
    def raw_train_dir(self) -> Path:
        return self.model_dir / BaseHub.RAW_TRAIN_DIR

    @cached_property
    def inference_dir(self) -> Path:
        return self.model_dir / BaseHub.INFERENCE_DIR

    @cached_property
    def evaluation_dir(self) -> Path:
        return self.model_dir / BaseHub.EVALUATION_DIR

    @cached_property
    def export_dir(self) -> Path:
        return self.model_dir / BaseHub.EXPORT_DIR

    @cached_property
    def train_option_file(self) -> Path:
        return self.train_dir / BaseHub.TRAIN_OPTION_FILE

    @cached_property
    def best_ckpt_file(self) -> Path:
        return self.train_dir / BaseHub.BEST_CKPT_FILE

    @cached_property
    def last_ckpt_file(self) -> Path:
        return self.train_dir / BaseHub.LAST_CKPT_FILE

    @cached_property
    def metric_file(self) -> Path:
        return self.train_dir / BaseHub.METRIC_FILE

    def delete_train(self):
        """Delete Raw Trained Data. It can be trained again."""
        io.remove_directory(self.raw_train_dir)

    def is_trainable(self):
        if self.raw_train_dir.exists():
            raise FileExistsError(
                f"""
                Train[{self.name}] already exists.
                Use another name or delete trains (hub.delete_train()).
                """
            )

    def check_train_sanity(self) -> bool:
        return (
            self.best_ckpt_file.exists()
            and self.last_ckpt_file.exists()
            and self.metric_file.exists()
        )

    @abstractmethod
    def validate_model(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        raise NotImplementedError

    @abstractmethod
    def evaluation(self):
        raise NotImplementedError

    @abstractmethod
    def export(self):
        raise NotImplementedError
