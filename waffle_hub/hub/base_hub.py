import logging
from abc import abstractmethod
from functools import cached_property
from pathlib import Path

from tabulate import tabulate
from waffle_utils.utils import type_validator

from waffle_hub import get_backends

logger = logging.getLogger(__name__)


class BaseHub:

    # directory settings
    DEFAULT_ROOT_DIR = Path("./models")

    TRAIN_DIR = Path("trains")
    INFERENCE_DIR = Path("inferences")
    EVALUATION_DIR = Path("evaluations")
    EXPORT_DIR = Path("exports")

    # common files
    LAST_CKPT_FILE = TRAIN_DIR / "weights/last_ckpt.pth"
    BEST_CKPT_FILE = TRAIN_DIR / "weights/best_ckpt.pth"

    # train files
    TRAIN_LOG_FILE = TRAIN_DIR / "log.txt"
    METRIC_FILE = TRAIN_DIR / "metrics.txt"

    # infer files
    INFERENCE_LOG_FILE = INFERENCE_DIR / "log.txt"

    # eval files
    EVALUATION_LOG_FILE = EVALUATION_DIR / "log.txt"

    # export files
    EXPORT_LOG_FILE = EXPORT_DIR / "log.txt"

    def __init__(
        self,
        name: str,
        root_dir: str = None,
        backend: str = None,
        version: str = None,
    ):

        self.name = self._name = name
        self.root_dir = self._root_dir = root_dir
        self.backend = self._backend = backend
        self.version = self._version = version

    # properties
    @property
    def name(self):
        return self._name

    @name.setter
    @type_validator(str)
    def name(self, v):
        self._name = v

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @root_dir.setter
    @type_validator(str)
    def root_dir(self, v):
        self._root_dir = Path(v) if v else BaseHub.DEFAULT_ROOT_DIR

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    @type_validator(str)
    def backend(self, v):
        backends = get_backends()
        if v is None:
            v = list(backends.keys())[0]
            logger.info(f"Using default backend {v}.")
        if v not in backends:
            raise ValueError(
                f"""
                Backend {v} is not supported.
                Choose one of {list(backends.keys())}
                """
            )
        self._backend = v

    @property
    def version(self) -> str:
        return self._version

    @version.setter
    @type_validator(str)
    def version(self, v):
        backends = get_backends()
        if v is None:
            v = backends[self.backend][-1]
            logger.info(f"Using default backend version {v}.")
        if v not in backends[self.backend]:
            raise ValueError(
                f"""
                Backend version {v} is not supported.
                Choose one of {backends[v]}
                """
            )
        self._version = v

    @cached_property
    def model_dir(self) -> Path:
        return self.root_dir / self.name

    @cached_property
    def train_dir(self) -> Path:
        return self.model_dir / BaseHub.TRAIN_DIR

    @cached_property
    def inference_dir(self) -> Path:
        return self.model_dir / BaseHub.INFERENCE_DIR

    @cached_property
    def evaluation_dir(self) -> Path:
        return self.model_dir / BaseHub.EVALUATION_DIR

    @cached_property
    def export_dir(self) -> Path:
        return self.model_dir / BaseHub.EXPORT_DIR

    @classmethod
    def get_available_backends(cls) -> str:
        """Available backends"""
        backends = get_backends()

        table_data = []
        for name, versions in backends.items():
            for i, version in enumerate(versions):
                table_data.append([name if i == 0 else "", version])

        table = str(
            tabulate(
                table_data,
                headers=["Backend", "Version"],
                tablefmt="simple_outline",
            )
        )
        return table

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
