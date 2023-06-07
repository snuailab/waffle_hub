import importlib
from pathlib import Path

from waffle_hub.hub.base_hub import BaseHub
from waffle_hub.schema.configs import ModelConfig

BACKEND_MAP = {
    "ultralytics": {
        "import_path": "waffle_hub.hub.adapter.ultralytics",
        "class_name": "UltralyticsHub",
    },
    "autocare_dlt": {
        "import_path": "waffle_hub.hub.adapter.autocare_dlt",
        "class_name": "AutocareDLTHub",
    },
    "transformers": {
        "import_path": "waffle_hub.hub.adapter.transformers",
        "class_name": "TransformersHub",
    },
}


def get_hub(backend: str) -> BaseHub:
    """
    Get hub

    Args:
        backend (str): Backend name

    Raises:
        ModuleNotFoundError: If backend is not supported

    Example:
        >>> from waffle_hub import get_hub
        >>> hub = get_hub("ultralytics").new(
        >>>     name="test",
        >>>     task="classification",
        >>>     model_type="yolov8",
        >>>     model_size="s",
        >>>     categories=["cat", "dog"],
        >>> )

    Returns:
        BaseHub: Backend hub Class
    """
    if backend not in BACKEND_MAP:
        raise ModuleNotFoundError(f"Backend {backend} is not supported")

    backend_info = BACKEND_MAP[backend]
    module = importlib.import_module(backend_info["import_path"])
    hub_class = getattr(module, backend_info["class_name"])
    return hub_class


def load_hub(name: str, root_dir: str = None) -> BaseHub:
    """Load Hub by name.

    Args:
        name (str): hub name.
        root_dir (str, optional): hub root directory. Defaults to None.

    Raises:
        FileNotFoundError: if hub is not exist in root_dir

    Returns:
        Hub: Hub instance
    """

    root_dir = Path(root_dir if root_dir else BaseHub.DEFAULT_ROOT_DIR)
    model_config_file = root_dir / name / BaseHub.MODEL_CONFIG_FILE
    if not model_config_file.exists():
        raise FileNotFoundError(f"Model[{name}] does not exists. {model_config_file}")
    model_config = ModelConfig.load(model_config_file)

    return get_hub(model_config.backend).load(name, root_dir)
