""" (.rst format docstring)
Hub
================
Hub is a multi-backend compatible interface for model training, evaluation, inference, and export.

.. note::
    Check out docstrings for more details.

"""
import importlib
import logging
import os
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import torch
import tqdm
from waffle_utils.file import io
from waffle_utils.validator import setter_type_validator

from waffle_hub import BACKEND_MAP, ExportWaffleStatus
from waffle_hub.dataset import Dataset
from waffle_hub.hub.eval.callbacks import EvaluateStateWriterCallback
from waffle_hub.hub.eval.evaluator import Evaluator
from waffle_hub.hub.infer.callbacks import (
    InferenceDrawCallback,
    InferenceShowCallback,
    InferenceStateWriterCallback,
)
from waffle_hub.hub.infer.inferencer import Inferencer
from waffle_hub.hub.manager.base_manager import BaseManager
from waffle_hub.hub.manager.callbacks import TrainStateWriterCallback
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.hub.onnx.callbacks import ExportOnnxStateWriterCallback
from waffle_hub.hub.onnx.exporter import OnnxExporter
from waffle_hub.schema.configs import ModelConfig, TrainConfig
from waffle_hub.schema.data import ImageInfo
from waffle_hub.schema.fields import Category
from waffle_hub.schema.result import (
    EvaluateResult,
    ExportOnnxResult,
    ExportWaffleResult,
    InferenceResult,
    TrainResult,
)
from waffle_hub.schema.state import ExportWaffleState, TrainState
from waffle_hub.type import BackendType, TaskType
from waffle_hub.utils.data import get_image_transform

logger = logging.getLogger(__name__)


class Hub:
    # directory settings
    DEFAULT_HUB_ROOT_DIR = Path("./hubs")

    # train files ##--
    TRAIN_CONFIG_FILE = BaseManager.CONFIG_DIR / BaseManager.TRAIN_CONFIG_FILE
    MODEL_CONFIG_FILE = BaseManager.CONFIG_DIR / BaseManager.MODEL_CONFIG_FILE

    # State files
    STATE_DIR = Path("states")
    TRAIN_STATE_FILE = STATE_DIR / "train_state.json"
    EVALUATE_STATE_FILE = STATE_DIR / "evaluate_state.json"
    EXPORT_ONNX_STATE_FILE = STATE_DIR / "export_onnx_state.json"
    EXPORT_WAFFLE_STATE_FILE = STATE_DIR / "export_waffle_state.json"
    INFERENCE_STATE_FILE = STATE_DIR / "inference_state.json"

    def __init__(
        self,
        name: str,
        backend: str = None,
        task: Union[str, TaskType] = None,
        model_type: str = None,
        model_size: str = None,
        categories: list[Union[str, int, float, dict, Category]] = None,
        root_dir: str = None,
        *args,
        **kwargs,
    ):

        self.root_dir: Path = root_dir

        self.name: str = name
        self.task: str = task
        self.model_type: str = model_type
        self.model_size: str = model_size
        self.categories: list[Category] = categories

        default_train_callbacks = [
            TrainStateWriterCallback(self.train_state_file, self.evaluate_state_file)
        ]

        train_state = self.get_train_state()
        self.manager = (
            self.get_manager_class(backend).load(
                root_dir=self.hub_dir,
                train_state=TrainState.from_dict(train_state) if train_state is not None else None,
                callbacks=default_train_callbacks,
            )
            if BaseManager.is_exists(root_dir=self.hub_dir)
            else self.get_manager_class(backend)(
                root_dir=self.hub_dir,
                name=self.name,
                task=self.task,
                model_type=self.model_type,
                model_size=self.model_size,
                categories=self.categories,
                callbacks=default_train_callbacks,
            )
        )
        if self.manager.name != self.name:
            self.manager.set_model_name(self.name)

        self.backend: str = self.manager.backend
        self.backend_version: str = self.manager.BACKEND_VERSION

    def __repr__(self):
        return self.get_model_config().__repr__()

    @classmethod
    def get_manager_class(cls, backend: str = None) -> "BaseManager":
        """
        Get training manager class

        Args:
            backend (str): Backend name

        Raises:
            ModuleNotFoundError: If backend is not supported

        Returns:
            BaseManager: Backend training manager Class
        """
        if backend not in list(BACKEND_MAP.keys()):
            raise ModuleNotFoundError(f"Backend {backend} is not supported")

        backend_info = BACKEND_MAP[backend]
        module = importlib.import_module(backend_info["adapter_import_path"])
        adapter_class = getattr(module, backend_info["adapter_class_name"])
        return adapter_class

    @classmethod
    def get_available_backends(cls) -> list[str]:
        """
        Get available backends

        Returns:
            list[str]: Available backends
        """
        return list(BACKEND_MAP.keys())

    @classmethod
    def get_available_tasks(cls, backend: str) -> list[str]:
        """
        Get available tasks

        Args:
            backend (str): Backend name

        Raises:
            ModuleNotFoundError: If backend is not supported

        Returns:
            list[str]: Available tasks
        """
        backend = backend if backend else cls.BACKEND_NAME
        manager = cls.get_manager_class(backend)
        return list(manager.MODEL_TYPES.keys())

    @classmethod
    def get_available_model_types(cls, backend: str, task: str) -> list[str]:
        """
        Get available model types

        Args:
            backend (str): Backend name
            task (str): Task name

        Raises:
            ModuleNotFoundError: If backend is not supported

        Returns:
            list[str]: Available model types
        """

        manager = cls.get_manager_class(backend)
        if task not in list(manager.MODEL_TYPES.keys()):
            raise ValueError(f"{task} is not supported with {backend}")
        task = TaskType.from_str(task).value
        return list(manager.MODEL_TYPES[task].keys())

    @classmethod
    def get_available_model_sizes(cls, backend: str, task: str, model_type: str) -> list[str]:
        """
        Get available model sizes

        Args:
            backend (str): Backend name
            task (str): Task name
            model_type (str): Model type

        Raises:
            ModuleNotFoundError: If backend is not supported

        Returns:
            list[str]: Available model sizes
        """
        manager = cls.get_manager_class(backend)
        if task not in list(manager.MODEL_TYPES.keys()):
            raise ValueError(f"{task} is not supported with {backend}")
        task = TaskType.from_str(task).value
        if model_type not in manager.MODEL_TYPES[task]:
            raise ValueError(f"{model_type} is not supported with {backend}")
        model_sizes = manager.MODEL_TYPES[task][model_type]
        return model_sizes if isinstance(model_sizes, list) else list(model_sizes.keys())

    @classmethod
    def get_default_train_params(
        cls, backend: str, task: str, model_type: str, model_size: str
    ) -> dict:
        """
        Get default train params

        Args:
            backend (str): Backend name
            task (str): Task name
            model_type (str): Model type
            model_size (str): Model size

        Raises:
            ModuleNotFoundError: If backend is not supported

        Returns:
            dict: Default train params
        """
        manager = cls.get_manager_class(backend)
        if task not in list(manager.MODEL_TYPES.keys()):
            raise ValueError(f"{task} is not supported with {backend}")
        task = TaskType.from_str(task).value
        if model_type not in manager.MODEL_TYPES[task]:
            raise ValueError(f"{model_type} is not supported with {backend}")
        if model_size not in manager.MODEL_TYPES[task][model_type]:
            raise ValueError(f"{model_size} is not supported with {backend}")
        return manager.DEFAULT_PARAMS[task][model_type][model_size]

    @classmethod
    def new(
        cls,
        name: str,
        backend: str = None,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        categories: Union[list[dict], list] = None,
        root_dir: str = None,
        *args,
        **kwargs,
    ) -> "Hub":
        """Create Hub.

        Args:
            name (str): Hub name
            backend (str, optional): Backend name. See Hub.get_available_backends. Defaults to None.
            task (str, optional): Task Name. See Hub.get_available_tasks. Defaults to None.
            model_type (str, optional): Model Type. See Hub.get_available_model_types. Defaults to None.
            model_size (str, optional): Model Size. See Hub.get_available_model_sizes. Defaults to None.
            categories (Union[list[dict], list], optional): class dictionary or list. [{"supercategory": "name"}, ] or ["name",]. Defaults to None.
            root_dir (str, optional): Root directory of hub repository. Defaults to None.

        Returns:
            Hub: Hub instance
        """
        root_dir = Hub.parse_root_dir(root_dir)

        if name in cls.get_hub_list(root_dir):
            raise FileExistsError(f"{name} already exists. Try another name.")

        try:
            backend = backend if backend else cls.get_available_backends()[0]
            task = TaskType.from_str(task).value if task else cls.get_available_tasks(backend)[0]
            model_type = (
                model_type if model_type else cls.get_available_model_types(backend, task)[0]
            )
            model_size = (
                model_size
                if model_size
                else cls.get_available_model_sizes(backend, task, model_type)[0]
            )

            return cls(
                name=name,
                backend=backend,
                task=task,
                model_type=model_type,
                model_size=model_size,
                categories=categories,
                root_dir=root_dir,
            )
        except Exception as e:
            if (root_dir / name).exists():
                io.remove_directory(root_dir / name, recursive=True)
            raise e

    @classmethod
    def load(cls, name: str, root_dir: str = None) -> "Hub":
        """Load Hub by name.

        Args:
            name (str): hub name.
            root_dir (str, optional): hub root directory. Defaults to None.

        Raises:
            FileNotFoundError: if hub is not exist in root_dir

        Returns:
            Hub: Hub instance
        """
        root_dir = Hub.parse_root_dir(root_dir)
        model_config_file = root_dir / name / BaseManager.CONFIG_DIR / BaseManager.MODEL_CONFIG_FILE
        if not model_config_file.exists():
            raise FileNotFoundError(f"Model[{name}] does not exists. {model_config_file}")
        model_config = ModelConfig.load(model_config_file)
        return cls(
            **{
                **model_config.to_dict(),
                "root_dir": root_dir,
            }
        )

    @classmethod
    def from_model_config(cls, name: str, model_config_file: str, root_dir: str = None) -> "Hub":
        """Create new Hub with model config.

        Args:
            name (str): hub name.
            model_config_file (str): model config yaml file.
            root_dir (str, optional): hub root directory. Defaults to None.

        Returns:
            Hub: New Hub instance
        """
        root_dir = Hub.parse_root_dir(root_dir)
        try:
            if name in cls.get_hub_list(root_dir):
                raise ValueError(f"{name} already exists. Try another name.")

            model_config = io.load_yaml(model_config_file)
            return cls.new(
                **{
                    **model_config,
                    "name": name,
                    "root_dir": root_dir,
                }
            )
        except Exception as e:
            if (root_dir / name).exists():
                io.remove_directory(root_dir / name, recursive=True)
            raise e

    @classmethod
    def get_hub_list(cls, root_dir: str = None) -> list[str]:
        """
        Get hub name list in root_dir.

        Args:
            root_dir (str, optional): hub root directory. Defaults to None.

        Returns:
            list[str]: hub name list
        """
        root_dir = Hub.parse_root_dir(root_dir)

        if not root_dir.exists():
            return []

        hub_name_list = []
        for hub_dir in root_dir.iterdir():
            if hub_dir.is_dir():
                model_config_file = hub_dir / BaseManager.CONFIG_DIR / BaseManager.MODEL_CONFIG_FILE
                if model_config_file.exists():
                    hub_name_list.append(hub_dir.name)
        return hub_name_list

    @classmethod
    def from_waffle_file(cls, name: str, waffle_file: str, root_dir: str = None) -> "Hub":
        """Import new Hub with waffle file for inference.

        Args:
            name (str): hub name.
            waffle_file (str): waffle file path.
            root_dir (str, optional): hub root directory. Defaults to None.

        Returns:
            Hub: New Hub instance
        """
        root_dir = Hub.parse_root_dir(root_dir)

        if name in cls.get_hub_list(root_dir):
            raise FileExistsError(f"{name} already exists. Try another name.")

        if not os.path.exists(waffle_file):
            raise FileNotFoundError(f"Waffle file {waffle_file} is not exist.")

        if os.path.splitext(waffle_file)[1] != ".waffle":
            raise ValueError(
                f"Invalid waffle file: {waffle_file}, Waffle File extension must be .waffle."
            )

        try:
            io.unzip(waffle_file, root_dir / name, create_directory=True)
            model_config_file = (
                root_dir / name / BaseManager.CONFIG_DIR / BaseManager.MODEL_CONFIG_FILE
            )
            if not model_config_file.exists():
                raise FileNotFoundError(
                    f"{model_config_file} does not exists. Please check waffle file."
                )
            model_config = io.load_yaml(model_config_file)
            return cls(
                **{
                    **model_config,
                    "name": name,
                    "root_dir": root_dir,
                }
            )

        except Exception as e:
            if (root_dir / name).exists():
                io.remove_directory(root_dir / name, recursive=True)
            raise e

    # properties
    @property
    def name(self) -> str:
        """Hub name"""
        return self.__name

    @name.setter
    @setter_type_validator(str)
    def name(self, v):
        self.__name = v

    @property
    def root_dir(self) -> Path:
        """Root Directory"""
        return self.__root_dir

    @root_dir.setter
    @setter_type_validator(Path, strict=False)
    def root_dir(self, v):
        self.__root_dir = Hub.parse_root_dir(v)
        logger.info(f"Hub root directory: {self.root_dir}")

    @classmethod
    def parse_root_dir(cls, v):
        if v:
            return Path(v)
        elif os.getenv("WAFFLE_HUB_ROOT_DIR", None):
            return Path(os.getenv("WAFFLE_HUB_ROOT_DIR"))
        else:
            return Hub.DEFAULT_HUB_ROOT_DIR

    @property
    def backend(self) -> str:
        """Backend name"""
        return self.__backend

    @backend.setter
    @setter_type_validator(str, strict=False)
    def backend(self, v):
        if v not in list(BACKEND_MAP.keys()):
            raise ValueError(
                f"Backend {v} is not supported. Choose one of {list(BACKEND_MAP.keys())}"
            )
        self.__backend = str(v.value) if isinstance(v, BackendType) else str(v)

    # path properties
    @property
    def hub_dir(self) -> Path:
        """Hub(Model) Directory"""
        return self.root_dir / self.name

    @property
    def state_dir(self) -> Path:
        """State Directory"""
        return self.hub_dir / self.STATE_DIR

    @property
    def waffle_file(self) -> Path:
        """Export Waffle file"""
        return self.hub_dir / f"{self.name}.waffle"

    # state
    @property
    def train_state_file(self) -> Path:
        """Train State Json File"""
        return self.hub_dir / self.TRAIN_STATE_FILE

    @property
    def evaluate_state_file(self) -> Path:
        """Evaluate State Json File"""
        return self.hub_dir / self.EVALUATE_STATE_FILE

    @property
    def export_onnx_state_file(self) -> Path:
        """Export ONNX State Json File"""
        return self.hub_dir / self.EXPORT_ONNX_STATE_FILE

    @property
    def export_waffle_state_file(self) -> Path:
        """Export Waffle State Json File"""
        return self.hub_dir / self.EXPORT_WAFFLE_STATE_FILE

    @property
    def inference_state_file(self) -> Path:
        """Inference State Json File"""
        return self.hub_dir / self.INFERENCE_STATE_FILE

    ## model
    @property
    def config_dir(self) -> Path:
        """Config Directory (model config, train config)"""
        return self.manager.config_dir

    @property
    def model_config_file(self) -> Path:
        """Model Config yaml File"""
        return self.manager.model_config_file

    ## trainer
    @property
    def weights_dir(self) -> Path:
        return self.manager.weights_dir

    @property
    def artifacts_dir(self) -> Path:
        """Artifact Directory. This is raw output of each backend."""
        return self.manager.artifacts_dir

    @property
    def train_config_file(self) -> Path:
        """Train Config yaml File"""
        return self.manager.train_config_file

    @property
    def best_ckpt_file(self) -> Path:
        """Best Checkpoint File"""
        return self.manager.best_ckpt_file

    @property
    def last_ckpt_file(self) -> Path:
        """Last Checkpoint File"""
        return self.manager.last_ckpt_file

    @property
    def metrics_file(self) -> Path:
        """Metrics File"""
        return self.manager.metric_file

    ## evaluator
    @property
    def evaluate_file(self) -> Path:
        """Evaluate Json File"""
        return self.hub_dir / Evaluator.EVALUATE_FILE

    ## inferencer
    @property
    def inference_dir(self) -> Path:
        """Inference Results Directory"""
        return self.hub_dir / Inferencer.INFERENCE_DIR

    @property
    def inference_file(self) -> Path:
        """Inference Results File"""
        return self.hub_dir / Inferencer.INFERENCE_FILE

    @property
    def draw_dir(self) -> Path:
        """Draw Results Directory"""
        return self.hub_dir / Inferencer.DRAW_DIR

    ## exporter
    @property
    def onnx_file(self) -> Path:
        """Best Checkpoint ONNX File"""
        return self.hub_dir / OnnxExporter.ONNX_FILE

    # getters
    def get_model_config(self) -> ModelConfig:
        """Get model config from model config file.

        Returns:
            ModelConfig: model config
        """
        return self.manager.get_model_config(root_dir=self.hub_dir)

    def get_train_config(self) -> TrainConfig:
        """Get train config from train config file.

        Returns:
            TrainConfig: train config
        """

        return self.manager.get_train_config(root_dir=self.hub_dir)

    def get_categories(self) -> list[Category]:
        return self.manager.categories

    def get_category_names(self) -> list[str]:
        return [category.name for category in self.manager.categories]

    def get_default_advance_train_params(
        self, task: str = None, model_type: str = None, model_size: str = None
    ) -> dict:
        return self.manager.get_default_advance_train_params(task, model_type, model_size)

    # get state
    def get_train_state(self) -> dict:
        """Get train state from train state file.

        Returns:
            dict: train state
        """
        if self.train_state_file.exists():
            return io.load_json(self.train_state_file)
        else:
            # TODO: Warning
            return None

    def get_evaluate_state(self) -> dict:
        """Get evaluate state from evaluate state file.

        Returns:
            dict: evaluate state
        """
        if self.evaluate_state_file.exists():
            return io.load_json(self.evaluate_state_file)
        else:
            # TODO: Warning
            return None

    def get_export_onnx_state(self) -> dict:
        """Get export onnx state from export onnx state file.

        Returns:
            dict: export onnx state
        """
        if self.export_onnx_state_file.exists():
            return io.load_json(self.export_onnx_state_file)
        else:
            return None

    def get_export_waffle_state(self) -> dict:
        """Get export waffle state from export waffle state file.

        Returns:
            dict: export waffle state
        """
        if self.export_waffle_state_file.exists():
            return io.load_json(self.export_waffle_state_file)
        else:
            return None

    def get_inference_state(self) -> dict:
        """Get inference state from inference state file.

        Returns:
            dict: inference state
        """
        if self.inference_state_file.exists():
            return io.load_json(self.inference_state_file)
        else:
            return None

    # get results
    def get_metrics(self) -> list[list[dict]]:
        """Get metrics per epoch from metric file.

        Example:
            >>> hub.get_metrics()
            [
                [
                    {
                        "tag": "epoch",
                        "value": "1",
                    },
                    {
                        "tag": "train_loss",
                        "value": "0.0012",
                    }
                ],
            ]

        Returns:
            list[dict]: metrics per epoch
        """
        return self.manager.get_metrics()
        # if not self.metric_file.exists(): ##--
        #     raise FileNotFoundError("Metric file is not exist. Train first!")

        # if not self.evaluate_file.exists():
        #     raise FileNotFoundError("Evaluate file is not exist. Train first!")

        # return io.load_json(self.metric_file)

    def get_evaluate_result(self) -> list[dict]:
        """Get evaluate result from evaluate file.

        Example:
            >>> hub.get_evaluate_result()
            [
                {
                    "tag": "mAP",
                    "value": 0.5,
                },
            ]

        Returns:
            list[dict]: evaluate result
        """
        return Evaluator.get_evaluate_result(root_dir=self.hub_dir)

    def get_inference_result(self) -> list[dict]:
        """Get inference result from inference file.

        Example:
            >>> hub.get_inference_result()
            [
                {
                    "id": "00000001",
                    "category": "person",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "score": 0.9,
                },
            ]

        Returns:
            list[dict]: inference result
        """
        return Inferencer.get_inference_result(root_dir=self.hub_dir)

    # common functions
    def delete_hub(self):
        """Delete Hub. Hub name can be used again."""
        self.manager.delete_manager()
        del self
        return None

    def delete_artifacts(self):
        """Delete Artifacts Directory. It can be trained again."""
        self.manager.delete_artifacts()
        if self.train_state_file.exists():
            io.remove_file(self.train_state_file)

    def check_train_sanity(self) -> bool:
        """Check if all essential files are exist.

        Returns:
            bool: True if all files are exist else False
        """
        return self.manager.check_train_sanity()

    def save_model_config(self):
        """Save ModelConfig."""
        self.manager.save_model_config(self.model_config_file)

    # Hub Utils
    def get_image_loader(self) -> tuple[torch.Tensor, ImageInfo]:
        """Get image loader function.

        Returns:
            tuple[torch.Tensor, ImageInfo]: input transform function

        Example:
            >>> transform = hub.get_image_loader()
            >>> image, image_info = transform("path/to/image.jpg")
            >>> model = hub.get_model()
            >>> output = model(image.unsqueeze(0))
        """
        train_config: TrainConfig = self.get_train_config()
        transform = get_image_transform(train_config.image_size, train_config.letter_box)

        def inner(x: Union[np.ndarray, str]):
            """Input Transform Function

            Args:
                x (Union[np.ndarray, str]): opencv image or image path

            Returns:
                tuple[torch.Tensor, ImageInfo]: image and image info
            """
            image, image_info = transform(x)
            return image, image_info

        return inner

    def get_model(self) -> ModelWrapper:
        return self.manager.get_model()

    def train(
        self,
        dataset: Union[Dataset, str, Path],
        dataset_root_dir: str = None,
        epochs: int = None,
        batch_size: int = None,
        image_size: Union[int, list[int]] = None,
        learning_rate: float = None,
        letter_box: bool = None,
        pretrained_model: str = None,
        device: str = "0",
        workers: int = 2,
        seed: int = 0,
        advance_params: Union[dict, str] = None,
        verbose: bool = True,
    ) -> TrainResult:
        """Start Train

        Args:
            dataset (Union[Dataset, str]): Waffle Dataset object or path or name.
            dataset_root_dir (str, optional): Waffle Dataset root directory. Defaults to None.
            epochs (int, optional): number of epochs. None to use default. Defaults to None.
            batch_size (int, optional): batch size. None to use default. Defaults to None.
            image_size (Union[int, list[int]], optional): image size. None to use default. Defaults to None.
            learning_rate (float, optional): learning rate. None to use default. Defaults to None.
            letter_box (bool, optional): letter box. None to use default. Defaults to None.
            pretrained_model (str, optional): pretrained model. None to use default. Defaults to None.
            device (str, optional):
                "cpu" or "gpu_id" or comma seperated "gpu_ids". Defaults to "0".
            workers (int, optional): number of workers. Defaults to 2.
            seed (int, optional): random seed. Defaults to 0.
            advance_params (Union[dict, str], optional): advance params dictionary or file (yaml, json) path. Defaults to None.
            verbose (bool, optional): verbose. Defaults to True.

        Raises:
            FileExistsError: if trained artifact exists.
            FileNotFoundError: if can not detect appropriate dataset.
            ValueError: if can not detect appropriate dataset.
            e: something gone wrong with ultralytics

        Example:
            >>> train_result = hub.train(
                    dataset=dataset,
                    epochs=100,
                    batch_size=16,
                    image_size=640,
                    learning_rate=0.001,
                    letterbox=False,
                    device="0",  # use gpu 0
                    # device="0,1,2,3",  # use gpu 0,1,2,3
                    # device="cpu",  # use cpu
                    workers=2,
                    seed=123
                )
            >>> train_result.best_ckpt_file
            hubs/my_hub/weights/best_ckpt.pt
            >>> train_result.metrics
            [[{"tag": "epoch", "value": 1}, {"tag": "train/loss", "value": 0.1}, ...], ...]

        Returns:
            TrainResult: train result
        """

        return self.manager.train(
            dataset=dataset,
            dataset_root_dir=dataset_root_dir,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=learning_rate,
            letter_box=letter_box,
            pretrained_model=pretrained_model,
            device=device,
            workers=workers,
            seed=seed,
            advance_params=advance_params,
            verbose=verbose,
        )

    # Evaluation
    def evaluate(
        self,
        dataset: Union[Dataset, str, Path],
        dataset_root_dir: str = None,
        set_name: str = "test",
        batch_size: int = 4,
        image_size: Union[int, list[int]] = None,
        letter_box: bool = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        half: bool = False,
        workers: int = 2,
        device: str = "0",
        draw: bool = False,
    ) -> EvaluateResult:
        """Start Evaluate

        Args:
            dataset (Union[Dataset, str]): Waffle Dataset object or path or name.
            dataset_root_dir (str, optional): Waffle Dataset root directory. Defaults to None.
            set_name (str, optional): Waffle Dataset evalutation set name. Defaults to "test".
            batch_size (int, optional): batch size. Defaults to 4.
            image_size (Union[int, list[int]], optional): image size. If None, use train config or defaults to 224.
            letter_box (bool, optional): letter box. If None, use train config or defaults to True.
            confidence_threshold (float, optional): confidence threshold. Not required in classification. Defaults to 0.25.
            iou_threshold (float, optional): iou threshold. Not required in classification. Defaults to 0.5.
            half (bool, optional): half. Defaults to False.
            workers (int, optional): workers. Defaults to 2.
            device (str, optional): device. Defaults to "0".
            draw (bool, optional): draw. Defaults to False.
            hold (bool, optional): hold. Defaults to True.

        Examples:
            >>> evaluate_result = hub.evaluate(
                    dataset=detection_dataset,
                    batch_size=4,
                    image_size=640,
                    letterbox=False,
                    confidence_threshold=0.25,
                    iou_threshold=0.5,
                    workers=4,
                    device="0",
                )
            # or you can use train option by passing None
            >>> evaluate_result = hub.evaluate(
                    ...
                    image_size=None,  # use train option or default to 224
                    letterbox=None,  # use train option or default to True
                    ...
                )
            >>> evaluate_result.metrics
            [{"tag": "mAP", "value": 0.1}, ...]

        Returns:
            EvaluateResult: evaluate result
        """
        evaluator = Evaluator(
            root_dir=self.hub_dir,
            model=self.manager.get_model(),
            callbacks=[EvaluateStateWriterCallback(save_path=self.evaluate_state_file)],
        )

        # config setting
        # overwrite training config
        train_config = self.get_train_config()
        if image_size is None and train_config is not None:
            image_size = train_config.image_size
        if letter_box is None and train_config is not None:
            letter_box = train_config.letter_box

        return evaluator.evaluate(
            dataset=dataset,
            dataset_root_dir=dataset_root_dir,
            set_name=set_name,
            batch_size=batch_size,
            image_size=image_size,
            letter_box=letter_box,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            half=half,
            workers=workers,
            device=device,
            draw=draw,
        )

    def inference(
        self,
        source: Union[str, Dataset],
        recursive: bool = True,
        image_size: Union[int, list[int]] = None,
        letter_box: bool = None,
        batch_size: int = 4,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        half: bool = False,
        workers: int = 2,
        device: str = "0",
        draw: bool = False,
        show: bool = False,
    ) -> InferenceResult:
        """Start Inference

        Args:
            source (str): image directory or image path or video path.
            recursive (bool, optional): recursive. Defaults to True.
            image_size (Union[int, list[int]], optional): image size. If None, use train config.
            letter_box (bool, optional): letter box. If None, use train config.
            batch_size (int, optional): batch size. Defaults to 4.
            confidence_threshold (float, optional): confidence threshold. Not required in classification. Defaults to 0.25.
            iou_threshold (float, optional): iou threshold. Not required in classification. Defaults to 0.5.
            half (bool, optional): half. Defaults to False.
            workers (int, optional): workers. Defaults to 2.
            device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
            draw (bool, optional): register draw callback. Defaults to False.
            show (bool, optional): register show callback. Defaults to False.


        Raises:
            FileNotFoundError: if can not detect appropriate dataset.
            e: something gone wrong with ultralytics

        Example:
            >>> inference_result = hub.inference(
                    source="path/to/images",
                    batch_size=4,
                    image_size=640,
                    letterbox=False,
                    confidence_threshold=0.25,
                    iou_threshold=0.5,
                    workers=4,
                    device="0",
                    draw=True,
                )
            # or simply use train option by passing None
            >>> inference_result = hub.inference(
                    ...
                    image_size=None,  # use train option or default to 224
                    letterbox=None,  # use train option or default to True
                    ...
                )
            >>> inference_result.predictions
            [{"relative/path/to/image/file": [{"category": "1", "bbox": [0, 0, 100, 100], "score": 0.9}, ...]}, ...]

        Returns:
            InferenceResult: inference result
        """
        inferencer = Inferencer(
            root_dir=self.hub_dir,
            model=self.manager.get_model(),
            callbacks=[InferenceStateWriterCallback(save_path=self.inference_state_file)],
        )
        # draw option
        if draw:
            inferencer.register_callback(InferenceDrawCallback(self.draw_dir))
        # show option
        if show:
            inferencer.register_callback(InferenceShowCallback())

        # overwrite training config
        train_config = self.get_train_config()
        if image_size is None and train_config is not None:
            image_size = train_config.image_size
        if letter_box is None and train_config is not None:
            letter_box = train_config.letter_box

        return inferencer.inference(
            source=source,
            recursive=recursive,
            image_size=image_size,
            letter_box=letter_box,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            half=half,
            workers=workers,
            device=device,
        )

    def benchmark(
        self,
        image_size: Union[int, list[int]] = None,
        batch_size: int = 16,
        device: str = "0",
        half: bool = False,
        trial: int = 100,
    ) -> dict:
        """Benchmark Model

        Args:
            image_size (Union[int, list[int]], optional): inference image size. None for same with train_config (recommended).
            batch_size (int, optional): dynamic batch size. Defaults to 16.
            device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
            half (bool, optional): half. Defaults to False.
            trial (int, optional): number of trials. Defaults to 100.

        Example:
            >>> hub.benchmark(
                    image_size=640,
                    batch_size=16,
                    device="0",
                    half=False,
                    trial=100,
                )
            {
                "inference_time": 0.123,
                "fps": 123.123,
                "image_size": [640, 640],
                "batch_size": 16,
                "device": "0",
                "cpu_name": "Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz",
                "gpu_name": "GeForce GTX 1080 Ti",
            }

        Returns:
            dict: benchmark result
        """
        inferencer = Inferencer(
            root_dir=self.hub_dir,
            model=self.manager.get_model(),
        )
        # overwrite training config or default
        train_config = self.get_train_config()
        if image_size is None and train_config is not None:
            image_size = train_config.image_size

        return inferencer.benchmark(
            image_size=image_size,
            batch_size=batch_size,
            device=device,
            half=half,
            trial=trial,
        )

    def export_onnx(
        self,
        image_size: Union[int, list[int]] = None,
        batch_size: int = 16,
        opset_version: int = 11,
        half: bool = False,
        device: str = "0",
    ) -> ExportOnnxResult:
        """Export Onnx Model

        Args:
            image_size (Union[int, list[int]], optional): image size. If None, same train config (recommended).
            batch_size (int, optional): dynamic batch size. Defaults to 16.
            opset_version (int, optional): onnx opset version. Defaults to 11.
            half (bool, optional): half. Defaults to False.
            device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
        Example:
            >>> export_onnx_result = hub.export_onnx(
                image_size=640,
                batch_size=16,
                opset_version=11,
            )
            # or simply use train option by passing None
            >>> export_onnx_result = hub.export_onnx(
                ...,
                image_size=None,  # use train option
                ...
            )
            >>> export_onnx_result.onnx_file
            hubs/my_hub/weights/model.onnx

        Returns:
            ExportOnnxResult: export onnx result
        """
        onnx_exporter = OnnxExporter(
            root_dir=self.hub_dir,
            model=self.manager.get_model(),
            callbacks=[ExportOnnxStateWriterCallback(save_path=self.export_onnx_state_file)],
        )

        # overwrite training config
        train_config = self.get_train_config()
        if image_size is None and train_config is not None:
            image_size = train_config.image_size

        return onnx_exporter.export(
            image_size=image_size,
            batch_size=batch_size,
            opset_version=opset_version,
            half=half,
            device=device,
        )

    def export_waffle(self) -> ExportWaffleResult:
        """Export Waffle Model
        Example:
            >>> export_waffle_result = hub.export_waffle()
            >>> export_waffle_result.waffle_file
            hubs/my_hub/my_hub.waffle
        Returns:
            ExportWaffleResult: export waffle result
        """
        self.check_train_sanity()

        try:
            state = ExportWaffleState(status=ExportWaffleStatus.INIT)
            state.save_json(self.export_waffle_state_file)
            result = ExportWaffleResult()

            io.zip([self.weights_dir, self.config_dir], self.waffle_file, recursive=True)
            result.waffle_file = self.waffle_file

            state.status = ExportWaffleStatus.SUCCESS
            state.save_json(self.export_waffle_state_file)

        except (KeyboardInterrupt, SystemExit) as e:
            state.status = ExportWaffleStatus.STOPPED
            state.save_json(self.export_waffle_state_file)
            if self.waffle_file.exists():
                io.remove_file(self.waffle_file)
        except Exception as e:
            state.status = ExportWaffleStatus.FAILED
            state.save_json(self.export_waffle_state_file)
            if self.waffle_file.exists():
                io.remove_file(self.waffle_file)
            raise e

        return result
