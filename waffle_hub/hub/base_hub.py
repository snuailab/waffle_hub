"""
Base Hub Class
Do not use this Class directly.
Use {Backend}Hub instead.
"""

import logging
from abc import abstractmethod
from dataclasses import asdict
from functools import cached_property
from pathlib import Path

from waffle_utils.file import io
from waffle_utils.utils import type_validator

from waffle_hub.schemas.configs import Model

logger = logging.getLogger(__name__)


class BaseHub:

    TASKS = [
        "object_detection",
        "classification",
        "segmentation",
        "keypoint_detection",
    ]
    MODEL_TYPES = []
    MODEL_SIZES = []

    # directory settings
    DEFAULT_ROOT_DIR = Path("./hubs")

    RAW_TRAIN_DIR = Path("artifacts")

    INFERENCE_DIR = Path("inferences")
    EVALUATION_DIR = Path("evaluations")
    EXPORT_DIR = Path("exports")

    # config files
    CONFIG_DIR = Path("configs")
    MODEL_CONFIG_FILE = CONFIG_DIR / "model.yaml"
    TRAIN_CONFIG_FILE = CONFIG_DIR / "train.yaml"
    CLASS_CONFIG_FILE = CONFIG_DIR / "classes.yaml"

    # train results
    LAST_CKPT_FILE = "weights/last_ckpt.pt"
    BEST_CKPT_FILE = "weights/best_ckpt.pt"  # TODO: best metric?
    METRIC_FILE = "metrics.csv"

    # export results
    ONNX_FILE = "weights/model.onnx"

    def __init__(
        self,
        name: str,
        backend: str = None,
        version: str = None,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        root_dir: str = None,
    ):

        self.name: str = name
        self.task: str = task
        self.model_type: str = model_type
        self.model_size: str = model_size
        self.root_dir: Path = Path(root_dir) if root_dir else None

        self.backend: str = backend
        self.version: str = version

        # save model config
        model_config = Model(
            name=self.name,
            backend=self.backend,
            version=self.version,
            task=self.task,
            model_type=self.model_type,
            model_size=self.model_size,
        )
        io.save_yaml(
            asdict(model_config),
            self.model_config_file,
            create_directory=True,
        )
        print(model_config)

    @classmethod
    def load(cls, name: str, root_dir: str = None) -> "BaseHub":
        """Load Hub by name.

        Args:
            name (str): hub name.
            root_dir (str, optional): hub root directory. Defaults to None.

        Raises:
            FileNotFoundError: if hub is not exist in root_dir

        Returns:
            Hub: Hub instance
        """
        model_config_file = (
            Path(root_dir if root_dir else BaseHub.DEFAULT_ROOT_DIR)
            / name
            / BaseHub.MODEL_CONFIG_FILE
        )
        if not model_config_file.exists():
            raise FileNotFoundError(
                f"Model[{name}] does not exists. {model_config_file}"
            )
        model_config = io.load_yaml(model_config_file)
        return cls(**model_config)

    @classmethod
    def from_model_config(
        cls, name: str, model_config_file: str, root_dir: str = None
    ) -> "BaseHub":
        """Create new Hub with model config.

        Args:
            name (str): hub name.
            model_config_file (str): model config yaml file.
            root_dir (str, optional): hub root directory. Defaults to None.

        Returns:
            Hub: New Hub instance
        """
        model_config = io.load_yaml(model_config_file)
        return cls(
            **{
                **model_config,
                "name": name,
                "root_dir": root_dir,
            }
        )

    # properties
    @property
    def name(self) -> str:
        """Hub name"""
        return self.__name

    @name.setter
    @type_validator(str)
    def name(self, v):
        self.__name = v

    @property
    def root_dir(self) -> Path:
        """Root Directory"""
        return self.__root_dir

    @root_dir.setter
    @type_validator(Path, strict=False)
    def root_dir(self, v):
        self.__root_dir = Path(v) if v else BaseHub.DEFAULT_ROOT_DIR

    @property
    def task(self) -> str:
        """Task Name"""
        return self.__task

    @task.setter
    @type_validator(str)
    def task(self, v):
        if v not in self.TASKS:
            raise ValueError(
                f"Task {v} is not supported. Choose one of {self.TASKS}"
            )
        self.__task = v

    @property
    def model_type(self) -> str:
        """Model Type"""
        return self.__model_type

    @model_type.setter
    @type_validator(str)
    def model_type(self, v):
        if v not in self.MODEL_TYPES:
            raise ValueError(
                f"Model Type {v} is not supported. Choose one of {self.MODEL_TYPES}"
            )
        self.__model_type = v

    @property
    def model_size(self) -> str:
        """Model Size"""
        return self.__model_size

    @model_size.setter
    @type_validator(str)
    def model_size(self, v):
        if v not in self.MODEL_SIZES:
            raise ValueError(
                f"Model Size {v} is not supported. Choose one of {self.MODEL_SIZES}"
            )
        self.__model_size = v

    @property
    def backend(self) -> str:
        """Backend name"""
        return self.__backend

    @backend.setter
    @type_validator(str)
    def backend(self, v):
        self.__backend = v

    @property
    def version(self) -> str:
        """Version"""
        return self.__version

    @version.setter
    @type_validator(str)
    def version(self, v):
        self.__version = v

    @cached_property
    def hub_dir(self) -> Path:
        """Hub(Model) Directory"""
        return self.root_dir / self.name

    @cached_property
    def artifact_dir(self) -> Path:
        """Artifact Directory. This is raw output of each backend."""
        return self.hub_dir / BaseHub.RAW_TRAIN_DIR

    @cached_property
    def inference_dir(self) -> Path:
        """Inference Results Directory"""
        return self.hub_dir / BaseHub.INFERENCE_DIR

    @cached_property
    def evaluation_dir(self) -> Path:
        """Evaluation Results Directory"""
        return self.hub_dir / BaseHub.EVALUATION_DIR

    @cached_property
    def export_dir(self) -> Path:
        """Export Results Directory"""
        return self.hub_dir / BaseHub.EXPORT_DIR

    @cached_property
    def model_config_file(self) -> Path:
        """Model Config yaml File"""
        return self.hub_dir / BaseHub.MODEL_CONFIG_FILE

    @cached_property
    def train_config_file(self) -> Path:
        """Train Config yaml File"""
        return self.hub_dir / BaseHub.TRAIN_CONFIG_FILE

    @cached_property
    def classes_config_file(self) -> Path:
        """Class Config yaml File"""
        return self.hub_dir / BaseHub.CLASS_CONFIG_FILE

    @cached_property
    def best_ckpt_file(self) -> Path:
        """Best Checkpoint File"""
        return self.hub_dir / BaseHub.BEST_CKPT_FILE

    @cached_property
    def onnx_file(self) -> Path:
        """Best Checkpoint File"""
        return self.hub_dir / BaseHub.ONNX_FILE

    @cached_property
    def last_ckpt_file(self) -> Path:
        """Last Checkpoint File"""
        return self.hub_dir / BaseHub.LAST_CKPT_FILE

    @cached_property
    def metric_file(self) -> Path:
        """Metric Csv File"""
        return self.hub_dir / BaseHub.METRIC_FILE

    def delete_artifact(self):
        """Delete Artifact Directory. It can be trained again."""
        io.remove_directory(self.artifact_dir)

    def check_train_sanity(self) -> bool:
        """Check if all essential files are exist.

        Returns:
            bool: True if all files are exist else False
        """
        return (
            self.classes_config_file.exists()
            and self.best_ckpt_file.exists()
            and self.last_ckpt_file.exists()
            and self.metric_file.exists()
        )

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
