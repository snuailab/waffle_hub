__version__ = "0.1.9"

import enum
import importlib
import warnings
import enum
from collections import OrderedDict

from tabulate import tabulate

# pytorch install check (necessary installation)
pytorch_versions = ["1.13.1"]
try:
    import torch

    if torch.__version__ not in pytorch_versions:
        warnings.warn(
            f"""
            torch {torch.__version__} has not been tested.
            We recommend you to use one of {pytorch_versions}
            """
        )
except ModuleNotFoundError as e:
    # TODO: Generalize install strings
    strings = []

    e.msg = "Need to install torch\n" + "\n".join(strings)
    raise e

# backend supports
_backends = OrderedDict(
    {
        "ultralytics": ["8.0.87"],
        "autocare_tx_model": ["0.2.0"],
        "transformers": ["4.27.4"],
    }
)


def get_backends() -> dict:
    return _backends


def get_installed_backend_version(backend: str) -> str:

    backends = get_backends()
    versions = backends[backend]

    if backend not in backends:
        raise ModuleNotFoundError(
            f"{backend} is not supported.\n Available backends {list(backends.keys())}"
        )

    try:
        module = importlib.import_module(backend)
        if module.__version__ not in versions:
            warnings.warn(
                f"""
                {backend} {module.__version__} has not been tested.
                We recommend you to use one of {versions}
                """
            )
        return module.__version__

    except ModuleNotFoundError as e:

        install_queries = "\n".join([f"- pip install {backend}=={version}" for version in versions])

        e.msg = f"""
            Need to install {backend}.
            Tested versions:
            {install_queries}
            """
        raise e


def get_available_backends() -> str:
    """Available backends"""
    backends = get_backends()

    table_data = []
    for name, versions in backends.items():
        for i, version in enumerate(versions):
            table_data.append([name if i == 0 else "", version])

    table = tabulate(
        table_data,
        headers=["Backend", "Version"],
        tablefmt="simple_outline",
    )

    return table

class CustomEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        if isinstance(item, str):
            return item.upper() in cls._member_names_
        return super().__contains__(item)

    def __upper__(self):
        return self.name.upper()


class BaseEnum(enum.Enum, metaclass=CustomEnumMeta):
    """Base class for Enum

    Example:
        >>> class Color(BaseEnum):
        >>>     RED = 1
        >>>     GREEN = 2
        >>>     BLUE = 3
        >>> Color.RED == "red"
        True
        >>> Color.RED == "RED"
        True
        >>> "red" in DataType
        True
        >>> "RED" in DataType
        True
    """

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name.upper() == other.upper()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.name.upper())

    def __str__(self):
        return self.name.upper()

    def __repr__(self):
        return self.name.upper()


class DataType(BaseEnum):
    # TODO: map to same value

    YOLO = enum.auto()
    ULTRALYTICS = enum.auto()

    COCO = enum.auto()
    TX_MODEL = enum.auto()
    AUTOCARE_TX_MODEL = enum.auto()

    HUGGINGFACE = enum.auto()
    TRANSFORMERS = enum.auto()


class TaskType(BaseEnum):
    CLASSIFICATION = enum.auto()
    OBJECT_DETECTION = enum.auto()
    SEGMENTATION = enum.auto()
    KEYPOINT_DETECTION = enum.auto()
    TEXT_RECOGNITION = enum.auto()
    REGRESSION = enum.auto()
