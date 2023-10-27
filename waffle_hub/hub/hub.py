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
import threading
import time
import warnings
from functools import cached_property
from pathlib import Path, PurePath
from typing import Union

import cpuinfo
import cv2
import numpy as np
import torch
import tqdm
from waffle_utils.file import io
from waffle_utils.image.io import save_image
from waffle_utils.utils import type_validator
from waffle_utils.video.io import create_video_writer

from waffle_dough.type.task_type import TaskType
from waffle_hub import BACKEND_MAP
from waffle_hub.dataset import Dataset
from waffle_hub.hub.model.result_parser import get_parser
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.hub.train.adapter.base_manager import BaseManager
from waffle_hub.schema.configs import (
    EvaluateConfig,
    ExportOnnxConfig,
    InferenceConfig,
    ModelConfig,
    TrainConfig,
)
from waffle_hub.schema.data import ImageInfo
from waffle_hub.schema.fields import Category
from waffle_hub.schema.result import (
    EvaluateResult,
    ExportOnnxResult,
    ExportWaffleResult,
    InferenceResult,
    TrainResult,
)
from waffle_hub.type.backend_type import BackendType
from waffle_hub.utils.callback import (
    EvaluateCallback,
    ExportCallback,
    InferenceCallback,
    TrainCallback,
)
from waffle_hub.utils.data import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    get_dataset_class,
    get_image_transform,
)
from waffle_hub.utils.draw import draw_results
from waffle_hub.utils.evaluate import evaluate_function
from waffle_hub.utils.memory import device_context
from waffle_hub.utils.metric_logger import MetricLogger

logger = logging.getLogger(__name__)


class Hub:
    # directory settings
    DEFAULT_HUB_ROOT_DIR = Path("./hubs")

    INFERENCE_DIR = Path("inferences")
    EXPORT_DIR = Path("exports")

    DRAW_DIR = Path("draws")

    # evaluate results
    EVALUATE_FILE = "evaluate.json"

    # inference results
    INFERENCE_FILE = "inferences.json"

    # export results
    ONNX_FILE = "weights/model.onnx"

    # train files ##--
    TRAIN_CONFIG_FILE = BaseManager.CONFIG_DIR / BaseManager.TRAIN_CONFIG_FILE
    MODEL_CONFIG_FILE = BaseManager.CONFIG_DIR / BaseManager.MODEL_CONFIG_FILE

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

        self.manager = self.get_manager_class(backend)(
            root_dir=self.hub_dir,
            name=self.name,
            task=self.task,
            model_type=self.model_type,
            model_size=self.model_size,
            categories=self.categories,
        )

        self.backend: str = self.manager.backend
        self.version: str = self.manager.VERSION

    def __repr__(self):
        return self.manager.get_model_config().__repr__()

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

    # @classmethod
    # def get_hub_class(cls, backend: str = None) -> "Hub":
    #     """
    #     Get hub class

    #     Args:
    #         backend (str): Backend name

    #     Raises:
    #         ModuleNotFoundError: If backend is not supported

    #     Returns:
    #         Hub: Backend hub Class
    #     """
    #     if backend not in BACKEND_MAP:
    #         raise ModuleNotFoundError(f"Backend {backend} is not supported")

    #     backend_info = BACKEND_MAP[backend]
    #     module = importlib.import_module(backend_info["import_path"])
    #     hub_class = getattr(module, backend_info["class_name"])
    #     return hub_class

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
        task = TaskType[task].value
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
        task = TaskType[task].value
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
        task = TaskType[task].value
        if model_type not in manager.MODEL_TYPES[task]:
            raise ValueError(f"{model_type} is not supported with {backend}")
        if model_size not in manager.MODEL_TYPES[task][model_type]:
            raise ValueError(f"{model_size} is not supported with {backend}")
        return manager.DEFAULT_PARAMS[task][model_type][model_size]

    @classmethod
    def test(cls):
        print(BaseManager.MODEL_CONFIG_FILE)

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
            task = TaskType[task].value if task else cls.get_available_tasks(backend)[0]
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
                io.remove_directory(root_dir / name)
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
                io.remove_directory(root_dir / name)
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
                raise FileNotFoundError(f"Model[{name}] does not exists. {model_config_file}")
            model_config = ModelConfig.load(model_config_file)
            model_config.name = name
            model_config.save_yaml(model_config_file)
            return cls(
                **{
                    **model_config.to_dict(),
                    "root_dir": root_dir,
                }
            )

        except Exception as e:
            if (root_dir / name).exists():
                io.remove_directory(root_dir / name)
            raise e

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
    @type_validator(str, strict=False)
    def backend(self, v):
        if v not in list(BACKEND_MAP.keys()):
            raise ValueError(
                f"Backend {v} is not supported. Choose one of {list(BACKEND_MAP.keys())}"
            )
        self.__backend = str(v.value) if isinstance(v, BackendType) else str(v)

    @cached_property
    def hub_dir(self) -> Path:
        """Hub(Model) Directory"""
        return self.root_dir / self.name

    @cached_property
    def inference_dir(self) -> Path:
        """Inference Results Directory"""
        return self.hub_dir / Hub.INFERENCE_DIR

    @cached_property
    def inference_file(self) -> Path:
        """Inference Results File"""
        return self.inference_dir / Hub.INFERENCE_FILE

    @cached_property
    def draw_dir(self) -> Path:
        """Draw Results Directory"""
        return self.inference_dir / Hub.DRAW_DIR

    @cached_property
    def onnx_file(self) -> Path:
        """Best Checkpoint ONNX File"""
        return self.hub_dir / Hub.ONNX_FILE

    @cached_property
    def evaluate_file(self) -> Path:
        """Evaluate Json File"""
        return self.hub_dir / Hub.EVALUATE_FILE

    @cached_property
    def waffle_file(self) -> Path:
        """Export Waffle file"""
        return self.hub_dir / f"{self.name}.waffle"

    # path getters
    ## model
    def get_model_config_file_path(self) -> Path:
        """Model Config yaml File"""
        return self.manager.model_config_file

    def get_artifact_dir(self) -> Path:
        """Artifact Directory. This is raw output of each backend."""
        return self.manager.artifact_dir

    def get_train_config_file_path(self) -> Path:
        """Train Config yaml File"""
        return self.manager.train_config_file

    def get_best_ckpt_file_path(self) -> Path:
        """Best Checkpoint File"""
        return self.manager.best_ckpt_file

    def get_last_ckpt_file_path(self) -> Path:
        """Last Checkpoint File"""
        return self.manager.last_ckpt_file

    def get_metrics_file_path(self) -> Path:
        """Metrics File"""
        return self.manager.metric_file

    # getters
    def get_model_config(self) -> ModelConfig:
        """Get model config from model config file.

        Returns:
            ModelConfig: model config
        """
        return self.manager.get_model_config()

    def get_train_config(self) -> TrainConfig:
        """Get train config from train config file.

        Returns:
            TrainConfig: train config
        """

        return self.manager.get_train_config()

    def get_categories(self) -> list[Category]:
        return self.manager.categories

    def get_category_names(self) -> list[str]:
        return [category.name for category in self.manager.categories]

    def get_default_advance_train_params(
        self, task: str = None, model_type: str = None, model_size: str = None
    ) -> dict:
        return self.manager.get_default_advance_train_params(task, model_type, model_size)

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
            dict: evaluate result
        """
        if not self.evaluate_file.exists():
            return []
        return io.load_json(self.evaluate_file)

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
        if not self.inference_file.exists():
            return []
        return io.load_json(self.inference_file)

    # common functions
    def delete_hub(self):
        """Delete all artifacts of Hub. Hub name can be used again."""
        io.remove_directory(self.hub_dir)
        del self
        return None

    def delete_artifact(self):
        """Delete Artifact Directory. It can be trained again."""
        self.manager.delete_artifact()

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

    def train(
        self,
        dataset: Union[Dataset, str],
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
        hold: bool = True,
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
            hold (bool, optional): hold process. Defaults to True.

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
            hold=hold,
        )

    # Evaluation Hook
    def get_model(self) -> ModelWrapper:
        return self.manager.get_model()

    def before_evaluate(self, cfg: EvaluateConfig, dataset: Dataset):
        if len(dataset.get_split_ids()[2]) == 0:
            cfg.set_name = "val"
            logger.warning("test set is not exist. use val set instead.")

    def on_evaluate_start(self, cfg: EvaluateConfig):
        pass

    def evaluating(self, cfg: EvaluateConfig, callback: EvaluateCallback, dataset: Dataset) -> str:
        device = cfg.device

        model = self.get_model().to(device)

        dataset = Dataset.load(cfg.dataset_name, cfg.dataset_root_dir)
        dataloader = get_dataset_class("dataset")(
            dataset,
            cfg.image_size,
            letter_box=cfg.letter_box,
            set_name=cfg.set_name,
        ).get_dataloader(cfg.batch_size, cfg.workers)

        result_parser = get_parser(self.task)(**cfg.to_dict(), categories=self.categories)

        callback._total_steps = len(dataloader) + 1

        preds = []
        labels = []
        for i, (images, image_infos, annotations) in tqdm.tqdm(
            enumerate(dataloader, start=1), total=len(dataloader)
        ):
            result_batch = model(images.to(device))
            result_batch = result_parser(result_batch, image_infos)

            preds.extend(result_batch)
            labels.extend(annotations)

            callback.update(i)

        metrics = evaluate_function(preds, labels, self.task, len(self.categories))

        result_metrics = []
        for tag, value in metrics.to_dict().items():
            if isinstance(value, list):
                values = [
                    {
                        "class_name": cat,
                        "value": cat_value,
                    }
                    for cat, cat_value in zip(self.get_category_names(), value)
                ]
            else:
                values = value
            result_metrics.append({"tag": tag, "value": values})
        io.save_json(result_metrics, self.evaluate_file)

    def on_evaluate_end(self, cfg: EvaluateConfig):
        pass

    def after_evaluate(self, cfg: EvaluateConfig, result: EvaluateResult):
        result.eval_metrics = self.get_evaluate_result()

    def evaluate(
        self,
        dataset: Union[Dataset, str],
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
        hold: bool = True,
    ) -> EvaluateResult:
        """Start Evaluate

        Args:
            dataset (Union[Dataset, str]): Waffle Dataset object or path or name.
            dataset_root_dir (str, optional): Waffle Dataset root directory. Defaults to None.
            batch_size (int, optional): batch size. Defaults to 4.
            image_size (Union[int, list[int]], optional): image size. Defaults to None.
            letter_box (bool, optional): letter box. Defaults to None.
            confidence_threshold (float, optional): confidence threshold. Defaults to 0.25.
            iou_threshold (float, optional): iou threshold. Defaults to 0.5.
            half (bool, optional): half. Defaults to False.
            workers (int, optional): workers. Defaults to 2.
            device (str, optional): device. Defaults to "0".
            draw (bool, optional): draw. Defaults to False.
            hold (bool, optional): hold. Defaults to True.

        Raises:
            FileNotFoundError: if can not detect appropriate dataset.
            e: something gone wrong with ultralytics

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
                    image_size=None,  # use train option
                    letterbox=None,  # use train option
                    ...
                )
            >>> evaluate_result.metrics
            [{"tag": "mAP", "value": 0.1}, ...]

        Returns:
            EvaluateResult: evaluate result
        """

        @device_context("cpu" if device == "cpu" else device)
        def inner(dataset: Dataset, callback: EvaluateCallback, result: EvaluateResult):
            try:
                self.before_evaluate(cfg, dataset)
                self.on_evaluate_start(cfg)
                self.evaluating(cfg, callback, dataset)
                self.on_evaluate_end(cfg)
                self.after_evaluate(cfg, result)
                callback.force_finish()
            except Exception as e:
                if self.evaluate_file.exists():
                    io.remove_file(self.evaluate_file)
                callback.force_finish()
                callback.set_failed()
                raise e

        if "," in device:
            warnings.warn("multi-gpu is not supported in evaluation. use first gpu only.")
            device = device.split(",")[0]

        if isinstance(dataset, (str, Path)):
            if Path(dataset).exists():
                dataset = Path(dataset)
                dataset = Dataset.load(
                    name=dataset.parts[-1], root_dir=dataset.parents[0].absolute()
                )
            elif dataset in Dataset.get_dataset_list(dataset_root_dir):
                dataset = Dataset.load(name=dataset, root_dir=dataset_root_dir)
            else:
                raise FileNotFoundError(f"Dataset {dataset} is not exist.")

        # overwrite training config
        train_config = self.get_train_config()
        if image_size is None:
            image_size = train_config.image_size
        if letter_box is None:
            letter_box = train_config.letter_box

        cfg = EvaluateConfig(
            dataset_name=dataset.name,
            set_name=set_name,
            batch_size=batch_size,
            image_size=image_size if isinstance(image_size, list) else [image_size, image_size],
            letter_box=letter_box,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            half=half,
            workers=workers,
            device="cpu" if device == "cpu" else f"cuda:{device}",
            draw=draw,
            dataset_root_dir=dataset.root_dir,
        )

        callback = EvaluateCallback(100)  # dummy step
        result = EvaluateResult()
        result.callback = callback

        if hold:
            inner(dataset, callback, result)
        else:
            thread = threading.Thread(target=inner, args=(dataset, callback, result), daemon=True)
            callback.register_thread(thread)
            callback.start()

        return result

    # inference hooks
    def before_inference(self, cfg: InferenceConfig):
        pass

    def on_inference_start(self, cfg: InferenceConfig):
        pass

    def inferencing(self, cfg: InferenceConfig, callback: InferenceCallback) -> str:
        device = cfg.device
        model = self.get_model().to(device)
        result_parser = get_parser(self.task)(**cfg.to_dict(), categories=self.categories)

        if cfg.source_type == "image":
            dataset = get_dataset_class(cfg.source_type)(
                cfg.source, cfg.image_size, letter_box=cfg.letter_box, recursive=cfg.recursive
            )
            dataloader = dataset.get_dataloader(cfg.batch_size, cfg.workers)
        elif cfg.source_type == "video":
            dataset = get_dataset_class(cfg.source_type)(
                cfg.source, cfg.image_size, letter_box=cfg.letter_box
            )
            dataloader = dataset.get_dataloader(cfg.batch_size, cfg.workers)
        else:
            raise ValueError(f"Invalid source type: {cfg.source_type}")

        if cfg.draw and cfg.source_type == "video":
            writer = None

        results = []
        callback._total_steps = len(dataloader) + 1
        for i, (images, image_infos) in tqdm.tqdm(
            enumerate(dataloader, start=1), total=len(dataloader)
        ):
            result_batch = model(images.to(device))
            result_batch = result_parser(result_batch, image_infos)
            for result, image_info in zip(result_batch, image_infos):

                results.append({str(image_info.image_rel_path): [res.to_dict() for res in result]})

                if cfg.draw:
                    io.make_directory(self.draw_dir)
                    draw = draw_results(
                        image_info.ori_image,
                        result,
                        names=[x["name"] for x in self.categories],
                    )

                    if cfg.source_type == "video":
                        if writer is None:
                            h, w = draw.shape[:2]
                            writer = create_video_writer(
                                str(self.inference_dir / Path(cfg.source).with_suffix(".mp4").name),
                                dataset.fps,
                                (w, h),
                            )
                        writer.write(draw)

                        draw_path = (
                            self.draw_dir
                            / Path(cfg.source).stem
                            / Path(image_info.image_rel_path).with_suffix(".png")
                        )
                    else:
                        draw_path = self.draw_dir / Path(image_info.image_rel_path).with_suffix(
                            ".png"
                        )
                    save_image(draw_path, draw, create_directory=True)

                if cfg.show:
                    if not cfg.draw:
                        draw = draw_results(
                            image_info.ori_image,
                            result,
                            names=[x["name"] for x in self.categories],
                        )
                    cv2.imshow("result", draw)
                    cv2.waitKey(1)

            callback.update(i)

        if cfg.draw and cfg.source_type == "video":
            writer.release()

        if cfg.show:
            cv2.destroyAllWindows()

        io.save_json(
            results,
            self.inference_file,
            create_directory=True,
        )

    def on_inference_end(self, cfg: InferenceConfig):
        pass

    def after_inference(self, cfg: InferenceConfig, result: EvaluateResult):
        result.predictions = self.get_inference_result()
        if cfg.draw:
            result.draw_dir = self.draw_dir

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
        hold: bool = True,
    ) -> InferenceResult:
        """Start Inference

        Args:
            source (str): image directory or image path or video path.
            recursive (bool, optional): recursive. Defaults to True.
            image_size (Union[int, list[int]], optional): image size. None for using training config. Defaults to None.
            letter_box (bool, optional): letter box. None for using training config. Defaults to None.
            batch_size (int, optional): batch size. Defaults to 4.
            confidence_threshold (float, optional): confidence threshold. Defaults to 0.25.
            iou_threshold (float, optional): iou threshold. Defaults to 0.5.
            half (bool, optional): half. Defaults to False.
            workers (int, optional): workers. Defaults to 2.
            device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
            draw (bool, optional): draw. Defaults to False.
            show (bool, optional): show. Defaults to False.
            hold (bool, optional): hold. Defaults to True.


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
                    image_size=None,  # use train option
                    letterbox=None,  # use train option
                    ...
                )
            >>> inference_result.predictions
            [{"relative/path/to/image/file": [{"category": "1", "bbox": [0, 0, 100, 100], "score": 0.9}, ...]}, ...]

        Returns:
            InferenceResult: inference result
        """

        @device_context("cpu" if device == "cpu" else device)
        def inner(callback: InferenceCallback, result: InferenceResult):
            try:
                self.before_inference(cfg)
                self.on_inference_start(cfg)
                self.inferencing(cfg, callback)
                self.on_inference_end(cfg)
                self.after_inference(cfg, result)
                callback.force_finish()
            except Exception as e:
                if self.inference_dir.exists():
                    io.remove_directory(self.inference_dir)
                callback.force_finish()
                callback.set_failed()
                raise e

        # image_dir, image_path, video_path, dataset_name, dataset
        if isinstance(source, (str, Path)):
            if Path(source).exists():
                source = Path(source)
                if source.is_dir():
                    source = source.absolute()
                    source_type = "image"
                elif source.suffix in IMAGE_EXTS:
                    source = [source.absolute()]
                    source_type = "image"
                elif source.suffix in VIDEO_EXTS:
                    source = str(source.absolute())
                    source_type = "video"
                else:
                    raise ValueError(
                        f"Invalid source: {source}\n"
                        + "Please use image directory or image path or video path."
                    )
            else:
                raise FileNotFoundError(f"Source {source} is not exist.")
        else:
            raise ValueError(
                f"Invalid source: {source}\n"
                + "Please use image directory or image path or video path."
            )

        # overwrite training config
        train_config = self.get_train_config()
        if image_size is None:
            image_size = train_config.image_size
        if letter_box is None:
            letter_box = train_config.letter_box

        cfg = InferenceConfig(
            source=source,
            source_type=source_type,
            batch_size=batch_size,
            recursive=recursive,
            image_size=image_size if isinstance(image_size, list) else [image_size, image_size],
            letter_box=letter_box,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            half=half,
            workers=workers,
            device="cpu" if device == "cpu" else f"cuda:{device}",
            draw=draw or show,
            show=show,
        )

        callback = InferenceCallback(100)  # dummy step
        result = InferenceResult()
        result.callback = callback
        if hold:
            inner(callback, result)
        else:
            thread = threading.Thread(target=inner, args=(callback, result), daemon=True)
            callback.register_thread(thread)
            callback.start()

        return result

    # Export Hook
    def before_export_onnx(self, cfg: ExportOnnxConfig):
        pass

    def on_export_onnx_start(self, cfg: ExportOnnxConfig):
        pass

    def exporting_onnx(self, cfg: ExportOnnxConfig, callback: ExportCallback) -> str:
        image_size = cfg.image_size
        image_size = [image_size, image_size] if isinstance(image_size, int) else image_size

        model = self.get_model().half() if cfg.half else self.get_model()
        model = model.to(cfg.device)

        input_name = ["inputs"]
        if self.task == TaskType.OBJECT_DETECTION:
            output_names = ["bbox", "conf", "class_id"]
        elif self.task == TaskType.CLASSIFICATION:
            output_names = ["predictions"]
        elif self.task == TaskType.INSTANCE_SEGMENTATION:
            output_names = ["bbox", "conf", "class_id", "masks"]
        elif self.task == TaskType.TEXT_RECOGNITION:
            output_names = ["class_ids", "confs"]
        else:
            raise NotImplementedError(f"{self.task} does not support export yet.")

        dummy_input = torch.randn(
            cfg.batch_size, 3, *image_size, dtype=torch.float16 if cfg.half else torch.float32
        )
        dummy_input = dummy_input.to(cfg.device)

        torch.onnx.export(
            model,
            dummy_input,
            str(self.onnx_file),
            input_names=input_name,
            output_names=output_names,
            opset_version=cfg.opset_version,
            dynamic_axes={name: {0: "batch_size"} for name in input_name + output_names},
        )

    def on_export_onnx_end(self, cfg: ExportOnnxConfig):
        pass

    def after_export_onnx(self, cfg: ExportOnnxConfig, result: ExportOnnxResult):
        result.onnx_file = self.onnx_file

    def export_onnx(
        self,
        image_size: Union[int, list[int]] = None,
        batch_size: int = 16,
        opset_version: int = 11,
        half: bool = False,
        device: str = "0",
        hold: bool = True,
    ) -> ExportOnnxResult:
        """Export Onnx Model

        Args:
            image_size (Union[int, list[int]], optional): inference image size. None for same with train_config (recommended).
            batch_size (int, optional): dynamic batch size. Defaults to 16.
            opset_version (int, optional): onnx opset version. Defaults to 11.
            half (bool, optional): half. Defaults to False.
            device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
            hold (bool, optional): hold or not.
                If True then it holds until task finished.
                If False then return Callback and run in background. Defaults to True.

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
        self.check_train_sanity()

        @device_context("cpu" if device == "cpu" else device)
        def inner(callback: ExportCallback, result: ExportOnnxResult):
            try:
                self.before_export_onnx(cfg)
                self.on_export_onnx_start(cfg)
                self.exporting_onnx(cfg, callback)
                self.on_export_onnx_end(cfg)
                self.after_export_onnx(cfg, result)
                callback.force_finish()
            except Exception as e:
                if self.onnx_file.exists():
                    io.remove_file(self.onnx_file)
                callback.force_finish()
                callback.set_failed()
                raise e

        # overwrite training config
        train_config = self.get_train_config()
        if image_size is None:
            image_size = train_config.image_size

        cfg = ExportOnnxConfig(
            image_size=image_size if isinstance(image_size, list) else [image_size, image_size],
            batch_size=batch_size,
            opset_version=opset_version,
            half=half,
            device="cpu" if device == "cpu" else f"cuda:{device}",
        )

        callback = ExportCallback(1)
        result = ExportOnnxResult()
        result.callback = callback

        if hold:
            inner(callback, result)
        else:
            thread = threading.Thread(target=inner, args=(callback, result), daemon=True)
            callback.register_thread(thread)
            callback.start()

        return result

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
        self.check_train_sanity()

        if half and (not torch.cuda.is_available() or device == "cpu"):
            raise RuntimeError("half is not supported in cpu")

        image_size = image_size or self.get_train_config().image_size
        image_size = [image_size, image_size] if isinstance(image_size, int) else image_size

        device = "cpu" if device == "cpu" else f"cuda:{device}"

        model = self.get_model()
        model = model.to(device) if not half else model.half().to(device)

        dummy_input = torch.randn(
            batch_size, 3, *image_size, dtype=torch.float32 if not half else torch.float16
        )
        dummy_input = dummy_input.to(device)

        model.eval()
        with torch.no_grad():
            start = time.time()
            for _ in tqdm.tqdm(range(trial)):
                model(dummy_input)
            end = time.time()
            inference_time = end - start

        del model

        return {
            "inference_time": inference_time,
            # image throughput per second
            "fps": trial * batch_size / inference_time,
            "image_size": image_size,
            "batch_size": batch_size,
            "precision": "fp16" if half else "fp32",
            "device": device,
            "cpu_name": cpuinfo.get_cpu_info()["brand_raw"],
            "gpu_name": torch.cuda.get_device_name(0) if device != "cpu" else None,
        }

    def export_waffle(self, hold: bool = True) -> ExportWaffleResult:
        """Export Waffle Model
        Args:
            hold (bool, optional): hold or not.
                If True then it holds until task finished.
                If False then return Callback and run in background. Defaults to True.
        Example:
            >>> export_waffle_result = hub.export_waffle()
            >>> export_waffle_result.waffle_file
            hubs/my_hub/my_hub.waffle
        Returns:
            ExportWaffleResult: export waffle result
        """
        self.check_train_sanity()

        def inner(callback: ExportCallback, result: ExportWaffleResult):
            try:
                io.zip(
                    [self.hub_dir / Hub.CONFIG_DIR, self.hub_dir / Hub.WEIGHTS_DIR], self.waffle_file
                )
                result.waffle_file = self.waffle_file
                callback.force_finish()
            except Exception as e:
                if self.waffle_file.exists():
                    io.remove_file(self.waffle_file)
                callback.force_finish()
                callback.set_failed()
                raise e

        callback = ExportCallback(1)
        result = ExportWaffleResult()
        result.callback = callback

        if hold:
            inner(callback, result)
        else:
            thread = threading.Thread(target=inner, args=(callback, result), daemon=True)
            callback.register_thread(thread)
            callback.start()

        return result
