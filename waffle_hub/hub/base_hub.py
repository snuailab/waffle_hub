import logging
from abc import abstractmethod
from dataclasses import asdict
from functools import cached_property
from pathlib import Path

from waffle_utils.file import io
from waffle_utils.utils import type_validator

from waffle_hub import get_backends
from waffle_hub.schemas.configs import Model

logger = logging.getLogger(__name__)


class BaseHub:

    TASKS = []
    MODEL_TYPES = []
    MODEL_SIZES = []

    # directory settings
    DEFAULT_ROOT_DIR = Path("./models")

    TRAIN_DIR = Path("trains")
    RAW_TRAIN_DIR = Path("raw_train")

    INFERENCE_DIR = Path("inferences")
    EVALUATION_DIR = Path("evaluations")
    EXPORT_DIR = Path("exports")

    # config files
    CONFIG_DIR = Path("configs")
    MODEL_CONFIG_FILE = CONFIG_DIR / "model.yaml"
    TRAIN_CONFIG_FILE = CONFIG_DIR / "train.yaml"
    CLASS_CONFIG_FILE = CONFIG_DIR / "classes.yaml"

    # train results
    LAST_CKPT_FILE = "weights/last_ckpt.pth"
    BEST_CKPT_FILE = "weights/best_ckpt.pth"
    METRIC_FILE = "metrics.txt"

    def __init__(
        self,
        name: str,
        backend: str,
        version: str = None,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        root_dir: str = None,
    ):

        self.name = name
        self.task = task
        self.model_type = model_type
        self.model_size = model_size
        self.backend = backend
        self.version = version
        self.root_dir = root_dir

        # save model config
        io.save_yaml(
            asdict(
                Model(
                    name=self.name,
                    backend=self.backend,
                    version=self.version,
                    task=self.task,
                    model_type=self.model_type,
                    model_size=self.model_size,
                )
            ),
            self.model_config_file,
            create_directory=True,
        )

    @classmethod
    def load(cls, name: str, root_dir: str = None):
        model_config_file = (
            Path(root_dir if root_dir else cls.DEFAULT_ROOT_DIR)
            / name
            / cls.MODEL_CONFIG_FILE
        )
        if not model_config_file.exists():
            raise FileNotFoundError(
                f"Model[{name}] does not exists. {model_config_file}"
            )
        return cls(**io.load_yaml(model_config_file))

    @classmethod
    def from_model_config(
        cls, name: str, model_config_file: str, root_dir: str = None
    ):
        return cls(
            **{
                **io.load_yaml(model_config_file),
                "name": name,
                "root_dir": root_dir,
            }
        )

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
        if v not in self.TASKS:
            raise ValueError(
                f"Task {v} is not supported. Choose one of {self.TASKS}"
            )
        self.__task = v

    @property
    def model_type(self) -> str:
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
        return self.__backend

    @backend.setter
    @type_validator(str)
    def backend(self, v):
        backends = list(get_backends().keys())
        if v not in backends:
            raise ValueError(
                f"Backend {v} is not supported. Choose one of {backends}"
            )
        self.__backend = v

    @property
    def version(self) -> str:
        return self.__version

    @version.setter
    @type_validator(str)
    def version(self, v):
        versions = get_backends()[self.backend]
        if v is None or v not in versions:
            v = versions[-1]
            logger.info(
                f"{self.backend} {v} is not supported. Using latest version {v}"
            )
        self.__version = v

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
    def model_config_file(self) -> Path:
        return self.model_dir / BaseHub.MODEL_CONFIG_FILE

    @cached_property
    def train_config_file(self) -> Path:
        return self.model_dir / BaseHub.TRAIN_CONFIG_FILE

    @cached_property
    def classes_config_file(self) -> Path:
        return self.model_dir / BaseHub.CLASS_CONFIG_FILE

    @cached_property
    def best_ckpt_file(self) -> Path:
        return self.model_dir / BaseHub.BEST_CKPT_FILE

    @cached_property
    def last_ckpt_file(self) -> Path:
        return self.model_dir / BaseHub.LAST_CKPT_FILE

    @cached_property
    def metric_file(self) -> Path:
        return self.model_dir / BaseHub.METRIC_FILE

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
