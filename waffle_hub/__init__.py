__version__ = "0.1.6"

import importlib
import warnings
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
    strings = [
        "  - conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch",
        "  - pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113",
        "  - pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1",
    ]

    e.msg = "Need to install torch\n" + "\n".join(strings)
    raise e

# backend supports
_backends = OrderedDict(
    {"ultralytics": ["8.0.72"], "autocare_tx_model": ["0.2.0"]}
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

        install_queries = "\n".join(
            [f"- pip install {backend}=={version}" for version in versions]
        )

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
