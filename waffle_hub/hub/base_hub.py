"""
Base Hub Class
Do not use this Class directly.
Use {Backend}Hub instead.
"""
import contextlib
import logging
import os
import sys
import threading
from abc import abstractmethod
from dataclasses import asdict, dataclass
from functools import cached_property
from io import StringIO
from pathlib import Path
from typing import Union

import cv2
import torch
import tqdm
from waffle_utils.file import io
from waffle_utils.utils import type_validator

from waffle_hub.hub.model.wrapper import get_parser
from waffle_hub.schemas.configs import Model, Train
from waffle_hub.utils.callback import (
    ExportCallback,
    InferenceCallback,
    TrainCallback,
)
from waffle_hub.utils.image import ImageDataset, draw_results

logger = logging.getLogger(__name__)


class ConfigContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return


@dataclass
class TrainContext(ConfigContext):
    dataset_path: str
    epochs: int
    batch_size: int
    image_size: int
    letter_box: bool
    pretrained_model: str
    device: str
    workers: int
    seed: int
    verbose: bool


@dataclass
class InferenceContext(ConfigContext):
    source: str
    batch_size: int
    recursive: bool
    image_size: int
    letter_box: bool
    confidence_threshold: float
    iou_threshold: float
    half: bool
    workers: int
    device: str
    draw: bool

    model = None
    dataloader = None


@dataclass
class ExportContext(ConfigContext):
    image_size: Union[int, list]
    batch_size: int
    input_name: list[str]
    output_name: list[str]
    opset_version: int


class BaseHub:

    MODEL_TYPES = {}

    # directory settings
    DEFAULT_ROOT_DIR = Path("./hubs")

    ARTIFACT_DIR = Path("artifacts")

    INFERENCE_DIR = Path("inferences")
    EVALUATION_DIR = Path("evaluations")
    EXPORT_DIR = Path("exports")

    DRAW_DIR = Path("draws")

    # config files
    CONFIG_DIR = Path("configs")
    MODEL_CONFIG_FILE = CONFIG_DIR / "model.yaml"
    TRAIN_CONFIG_FILE = CONFIG_DIR / "train.yaml"

    # train results
    LAST_CKPT_FILE = "weights/last_ckpt.pt"
    BEST_CKPT_FILE = "weights/best_ckpt.pt"  # TODO: best metric?
    METRIC_FILE = "metrics.json"

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
        classes: Union[list[dict], list] = None,
        root_dir: str = None,
    ):
        self.name: str = name
        self.task: str = task
        self.model_type: str = model_type
        self.model_size: str = model_size
        self.classes: list[dict] = classes
        self.root_dir: Path = Path(root_dir) if root_dir else None

        self.backend: str = backend
        self.version: str = version

        # check task supports
        if self.task not in self.MODEL_TYPES:
            io.remove_directory()
            raise ValueError(
                f"{self.task} is not supported with {self.backend}"
            )

        try:
            # save model config
            model_config = Model(
                name=self.name,
                backend=self.backend,
                version=self.version,
                task=self.task,
                model_type=self.model_type,
                model_size=self.model_size,
                classes=self.classes,
            )
            io.save_yaml(
                asdict(model_config),
                self.model_config_file,
                create_directory=True,
            )
        except Exception as e:
            raise e

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
        if v not in self.MODEL_TYPES:
            raise ValueError(
                f"Task {v} is not supported. Choose one of {self.MODEL_TYPES}"
            )
        self.__task = v

    @property
    def model_type(self) -> str:
        """Model Type"""
        return self.__model_type

    @model_type.setter
    @type_validator(str)
    def model_type(self, v):
        if v not in self.MODEL_TYPES[self.task]:
            raise ValueError(
                f"Model Type {v} is not supported. Choose one of {self.MODEL_TYPES[self.task]}"
            )
        self.__model_type = v

    @property
    def model_size(self) -> str:
        """Model Size"""
        return self.__model_size

    @model_size.setter
    @type_validator(str)
    def model_size(self, v):
        if v not in self.MODEL_TYPES[self.task][self.model_type]:
            raise ValueError(
                f"Model Size {v} is not supported. Choose one of {self.MODEL_TYPES[self.task][self.model_type]}"
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

    @property
    def classes(self) -> list[dict]:
        return self.__classes

    @classes.setter
    @type_validator(list)
    def classes(self, v):
        if isinstance(v[0], str):
            v = [{"supercategory": "object", "name": n} for n in v]
        self.__classes = v

    @cached_property
    def hub_dir(self) -> Path:
        """Hub(Model) Directory"""
        return self.root_dir / self.name

    @cached_property
    def artifact_dir(self) -> Path:
        """Artifact Directory. This is raw output of each backend."""
        return self.hub_dir / BaseHub.ARTIFACT_DIR

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
    def draw_dir(self) -> Path:
        """Draw Results Directory"""
        return self.hub_dir / BaseHub.DRAW_DIR

    @cached_property
    def model_config_file(self) -> Path:
        """Model Config yaml File"""
        return self.hub_dir / BaseHub.MODEL_CONFIG_FILE

    @cached_property
    def train_config_file(self) -> Path:
        """Train Config yaml File"""
        return self.hub_dir / BaseHub.TRAIN_CONFIG_FILE

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

    # common functions
    def delete_artifact(self):
        """Delete Artifact Directory. It can be trained again."""
        io.remove_directory(self.artifact_dir)

    def check_train_sanity(self) -> bool:
        """Check if all essential files are exist.

        Returns:
            bool: True if all files are exist else False
        """
        if not (
            self.model_config_file.exists()
            and self.best_ckpt_file.exists()
            # and self.last_ckpt_file.exists()
        ):
            raise FileNotFoundError("Train first! hub.train(...).")
        return True

    # Train Hook
    def before_train(self, ctx: TrainContext):
        if self.artifact_dir.exists():
            raise FileExistsError(
                f"{self.artifact_dir}\n"
                "Train artifacts already exist. Remove artifact to re-train (hub.delete_artifact())."
            )

    def on_train_start(self, ctx: TrainContext):
        pass

    def save_train_config(self, ctx: TrainContext):
        io.save_yaml(
            asdict(
                Train(
                    image_size=ctx.image_size,
                    letter_box=ctx.letter_box,
                    batch_size=ctx.batch_size,
                    pretrained_model=ctx.pretrained_model,
                    seed=ctx.seed,
                )
            ),
            self.train_config_file,
            create_directory=True,
        )

    def training(self, ctx: TrainContext):
        pass

    def on_train_end(self, ctx: TrainContext):
        pass

    def after_train(self, ctx: TrainContext):
        pass

    def train(
        self,
        dataset_path: str,
        epochs: int,
        batch_size: int,
        image_size: int,
        letter_box: bool = False,
        pretrained_model: str = None,
        device: str = "0",
        workers: int = 2,
        seed: int = 0,
        verbose: bool = True,
        hold: bool = True,
    ) -> str:
        """Start Train

        Args:
            dataset_path (str): Dataset Path. Recommend to use result of waffle_utils.dataset.Dataset.export.
            epochs (int): total epochs
            batch_size (int): batch size
            image_size (int): image size
            letter_box (bool): letter box preprocess. Defaults to False.
            pretrained_model (str, optional): pretrained model file. Defaults to None.
            device (str, optional): gpu device. Defaults to "0".
            workers (int, optional): num workers. Defaults to 2.
            seed (int, optional): random seed. Defaults to 0.
            verbose (bool, optional): verbose. Defaults to True.
            hold (bool, optional): hold or not.
                If True then it holds until task finished.
                If False then return Inferece Callback and run in background. Defaults to True.

        Raises:
            FileExistsError: if trained artifact exists.
            FileNotFoundError: if can not detect appropriate dataset.
            ValueError: if can not detect appropriate dataset.
            e: something gone wrong with ultralytics

        Returns:
            str: hub directory
        """

        ctx = TrainContext(
            dataset_path=dataset_path,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            letter_box=letter_box,
            pretrained_model=pretrained_model,
            device=device,
            workers=workers,
            seed=seed,
            verbose=verbose,
        )
        self.before_train(ctx)
        self.on_train_start(ctx)
        self.save_train_config(ctx)

        def inner(callback: TrainCallback):
            try:
                self.training(ctx, callback)
                callback.best_ckpt_file = self.best_ckpt_file
                callback.last_ckpt_file = self.last_ckpt_file
                callback.metric_file = self.metric_file
                callback.result_dir = self.hub_dir
                callback.force_finish()
                self.on_train_end(ctx)
                self.after_train(ctx)
            except Exception as e:
                if self.artifact_dir.exists():
                    io.remove_directory(self.artifact_dir)
                    callback.force_finish()
                raise e

        callback = TrainCallback(ctx.epochs, self.get_metrics)
        if hold:
            inner(callback)
        else:
            thread = threading.Thread(
                target=inner, args=(callback,), daemon=True
            )
            callback.register_thread(thread)
            callback.start()

        return callback

    # Inference Hook
    def get_model(self):
        raise NotImplementedError

    def before_inference(self, ctx: InferenceContext):
        self.check_train_sanity()

        # overwrite training config
        train_config = io.load_yaml(self.train_config_file)
        if ctx.image_size is None:
            ctx.image_size = train_config.get("image_size")
        if ctx.letter_box is None:
            ctx.letter_box = train_config.get("letter_box")

    def on_inference_start(self, ctx: InferenceContext):
        ctx.model = self.get_model(
            ctx.image_size, get_parser(self.task)(**asdict(ctx))
        )
        ctx.dataloader = ImageDataset(
            ctx.source, ctx.image_size, letter_box=ctx.letter_box
        ).get_dataloader(ctx.batch_size, ctx.workers)

    def inferencing(
        self, ctx: InferenceContext, callback: InferenceCallback
    ) -> str:
        model = ctx.model.to(ctx.device)
        dataloader = ctx.dataloader
        device = ctx.device

        for i, (images, image_infos) in tqdm.tqdm(
            enumerate(dataloader, start=1), total=len(dataloader)
        ):
            result_batch = model(images.to(device), image_infos)
            results = []
            for result, image_info in zip(result_batch, image_infos):
                image_path = image_info.get("image_path")

                relpath = Path(image_path).relative_to(ctx.source)
                io.save_json(
                    result,
                    self.inference_dir / relpath.with_suffix(".json"),
                    create_directory=True,
                )

                results.append(result)

                if ctx.draw:
                    draw = draw_results(
                        image_path,
                        result,
                        task=self.task,
                        names=[x["name"] for x in self.classes],
                    )
                    draw_path = self.draw_dir / relpath.with_suffix(".png")
                    io.make_directory(draw_path.parent)
                    cv2.imwrite(str(draw_path), draw)

            callback.update(i)

    def on_inference_end(self, ctx: InferenceContext):
        pass

    def after_inference(self, ctx: InferenceContext):
        pass

    def inference(
        self,
        source: str,
        recursive: bool = True,
        image_size: int = None,
        letter_box: bool = None,
        batch_size: int = 4,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        half: bool = False,
        workers: int = 2,
        device: str = "0",
        draw: bool = False,
        hold: bool = True,
    ) -> str:
        """Start Inference

        Args:
            source (str): dataset source. image file or image directory. TODO: video
            recursive (bool, optional): get images from directory recursively. Defaults to True.
            image_size (int, optional): inference image size. None for same with train_config (recommended).
            letter_box (bool, optional): letter box preprocess. None for same with train_config (recommended).
            batch_size (int, optional): batch size. Defaults to 4.
            confidence_threshold (float, optional): confidence threshold. Defaults to 0.25.
            iou_threshold (float, optional): iou threshold. Defaults to 0.7.
            half (bool, optional): fp16 inference. Defaults to False.
            device (str, optional): gpu device. Defaults to "0".
            draw (bool, optional): save draw or not. Defaults to False.
            hold (bool, optional): hold or not.
                If True then it holds until task finished.
                If False then return Inferece Callback and run in background. Defaults to True.

        Raises:
            FileNotFoundError: if can not detect appropriate dataset.
            e: something gone wrong with ultralytics

        Returns:
            str: inference result directory
        """
        self.check_train_sanity()

        ctx = InferenceContext(
            source=source,
            batch_size=batch_size,
            recursive=recursive,
            image_size=image_size,
            letter_box=letter_box,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            half=half,
            workers=workers,
            device="cpu" if device == "cpu" else f"cuda:{device}",
            draw=draw,
        )

        self.before_inference(ctx)
        self.on_inference_start(ctx)

        def inner(callback):
            try:
                self.inferencing(ctx, callback)
                callback.inference_dir = self.inference_dir
                callback.draw_dir = self.draw_dir if ctx.draw else None
                callback.force_finish()
                self.on_inference_end(ctx)
                self.after_inference(ctx)
            except Exception as e:
                if self.inference_dir.exists():
                    io.remove_directory(self.inference_dir)
                callback.force_finish()
                raise e

        callback = InferenceCallback(len(ctx.dataloader))

        if hold:
            inner(callback)
        else:
            thread = threading.Thread(
                target=inner, args=(callback,), daemon=True
            )
            callback.register_thread(thread)
            callback.start()

        return callback

    # Export Hook
    def export(
        self,
        image_size: int = None,
        batch_size: int = 1,
        opset_version: int = 11,
        hold: bool = True,
    ) -> str:
        """Export Model

        Args:
            image_size (int, optional): inference image size. None for same with train_config (recommended).
            batch_size (int, optional): dynamic batch size. Defaults to 16.
            opset_version (int, optional): onnx opset version. Defaults to 11.
            hold (bool, optional): hold or not.
                If True then it holds until task finished.
                If False then return Inferece Callback and run in background. Defaults to True.

        Returns:
            str: export onnx file path
        """
        self.check_train_sanity()

        train_config = Train(**io.load_yaml(self.train_config_file))

        image_size = image_size if image_size else train_config.image_size
        image_size = (
            [image_size, image_size]
            if isinstance(image_size, int)
            else image_size
        )

        model = self.get_model(train_config.image_size)

        input_name = ["inputs"]
        if self.task == "object_detection":
            output_names = ["bbox", "conf", "class_id"]
        elif self.task == "classification":
            output_names = ["predictions"]
        else:
            raise NotImplementedError(
                f"{self.task} does not support export yet."
            )

        dummy_input = torch.randn(batch_size, 3, *image_size)

        def inner(callback):
            try:
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(self.onnx_file),
                    input_names=input_name,
                    output_names=output_names,
                    opset_version=opset_version,
                    dynamic_axes={
                        name: {0: "batch_size"}
                        for name in input_name + output_names
                    },
                )
                callback.export_file = self.onnx_file
                callback.force_finish()
            except Exception as e:
                if self.onnx_file.exists():
                    io.remove_file(self.onnx_file)
                callback.force_finish()
                raise e

        callback = ExportCallback(1)

        if hold:
            inner(callback)
        else:
            thread = threading.Thread(
                target=inner, args=(callback,), daemon=True
            )
            callback.register_thread(thread)
            callback.start()

        return callback

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError
