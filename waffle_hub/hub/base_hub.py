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
        self.root_dir = self._root_dir = (
            Path(root_dir) if root_dir else BaseHub.DEFAULT_ROOT_DIR
        )

        backends = get_backends()
        if backend is None:
            backend = list(backends.keys())[0]
            logger.info(f"Using default backend {backend}.")

        if version is None:
            version = backends[backend][-1]
            logger.info(f"Using default version {version}.")

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
    def root_dir(self):
        return self._root_dir

    @root_dir.setter
    @type_validator(Path)
    def root_dir(self, v):
        self._root_dir = v

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

    @classmethod
    def is_available_backend(cls, name: str, version: str) -> bool:
        """Check if backend is available

        Args:
            name (str): backend name
            version (str): backend version

        Returns:
            bool: is available?
        """
        backends = get_backends()

        return (name in backends) and (version in backends[name])

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def remove(self):
        raise NotImplementedError

    @abstractmethod
    def get_progress(self):
        raise NotImplementedError

    @abstractmethod
    def get_status(self):
        raise NotImplementedError

    @abstractmethod
    def get_results(self):
        raise NotImplementedError

    @abstractmethod
    def get_log(self):
        raise NotImplementedError
