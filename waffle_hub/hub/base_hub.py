""" (.rst format docstring)
Hub
================
Hub is a multi-backend compatible interface for model training, evaluation, inference, and export.

.. note::
    Check out docstrings for more details.

Advanced Usage using threads
----------------

.. code-block:: python
    import time

    result = hub.some_job(..., hold=False)

    while (
        not result.callback.is_finished()
        and not result.callback.is_failed()
    ):
        time.sleep(1)
        print(result.callback.get_progress())

"""
import logging
import threading
import warnings
from functools import cached_property
from pathlib import Path
from typing import Union

import cv2
import torch
import tqdm
from waffle_utils.file import io
from waffle_utils.utils import type_validator

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub.model.wrapper import get_parser
from waffle_hub.schema.configs import (
    EvaluateConfig,
    ExportConfig,
    InferenceConfig,
    ModelConfig,
    TrainConfig,
)
from waffle_hub.schema.result import (
    EvaluateResult,
    ExportResult,
    InferenceResult,
    TrainResult,
)
from waffle_hub.utils.callback import (
    EvaluateCallback,
    ExportCallback,
    InferenceCallback,
    TrainCallback,
)
from waffle_hub.utils.data import ImageDataset, LabeledDataset
from waffle_hub.utils.draw import draw_results
from waffle_hub.utils.evaluate import evaluate_function

logger = logging.getLogger(__name__)


class BaseHub:

    MODEL_TYPES = {}

    # directory settings
    DEFAULT_ROOT_DIR = Path("./hubs")

    ARTIFACT_DIR = Path("artifacts")

    INFERENCE_DIR = Path("inferences")
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

    # evaluate results
    EVALUATE_FILE = "evaluate.json"

    # inference results
    INFERENCE_FILE = "inferences.json"

    # export results
    ONNX_FILE = "weights/model.onnx"

    def __init__(
        self,
        name: str,
        backend: str = None,
        version: str = None,
        task: Union[str, TaskType] = None,
        model_type: str = None,
        model_size: str = None,
        categories: Union[list[dict], list] = None,
        root_dir: str = None,
    ):
        self.name: str = name
        self.task: str = task
        self.model_type: str = model_type
        self.model_size: str = model_size
        self.categories: list[dict] = categories
        self.root_dir: Path = Path(root_dir) if root_dir else None

        self.backend: str = backend
        self.version: str = version

        # check task supports
        if self.task not in self.MODEL_TYPES:
            io.remove_directory()
            raise ValueError(f"{self.task} is not supported with {self.backend}")

        try:
            # save model config
            model_config = ModelConfig(
                name=self.name,
                backend=self.backend,
                version=self.version,
                task=self.task,
                model_type=self.model_type,
                model_size=self.model_size,
                categories=self.categories,
            )
            model_config.save_yaml(self.model_config_file)
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
        root_dir = Path(root_dir if root_dir else BaseHub.DEFAULT_ROOT_DIR)
        model_config_file = root_dir / name / BaseHub.MODEL_CONFIG_FILE
        if not model_config_file.exists():
            raise FileNotFoundError(f"Model[{name}] does not exists. {model_config_file}")
        model_config = io.load_yaml(model_config_file)
        return cls(
            **{
                **model_config,
                "root_dir": root_dir,
            }
        )

    @classmethod
    def from_model_config(cls, name: str, model_config_file: str, root_dir: str = None) -> "BaseHub":
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
    def task(self, v):
        v = str(v).lower()  # TODO: MODEL_TYPES should be enum
        if v not in self.MODEL_TYPES:
            raise ValueError(f"Task {v} is not supported. Choose one of {self.MODEL_TYPES}")
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
    def categories(self) -> list[dict]:
        return self.__categories

    @categories.setter
    @type_validator(list)
    def categories(self, v):
        if isinstance(v[0], str):
            v = [{"supercategory": "object", "name": n} for n in v]
        self.__categories = v

    @cached_property
    def hub_dir(self) -> Path:
        """Hub(Model) Directory"""
        return self.root_dir / self.name

    @cached_property
    def model_config_file(self) -> Path:
        """Model Config yaml File"""
        return self.hub_dir / BaseHub.MODEL_CONFIG_FILE

    @cached_property
    def artifact_dir(self) -> Path:
        """Artifact Directory. This is raw output of each backend."""
        return self.hub_dir / BaseHub.ARTIFACT_DIR

    @cached_property
    def inference_dir(self) -> Path:
        """Inference Results Directory"""
        return self.hub_dir / BaseHub.INFERENCE_DIR

    @cached_property
    def inference_file(self) -> Path:
        """Inference Results File"""
        return self.inference_dir / BaseHub.INFERENCE_FILE

    @cached_property
    def draw_dir(self) -> Path:
        """Draw Results Directory"""
        return self.inference_dir / BaseHub.DRAW_DIR

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

    @cached_property
    def evaluate_file(self) -> Path:
        """Evaluate Json File"""
        return self.hub_dir / BaseHub.EVALUATE_FILE

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

    def get_train_config(self) -> TrainConfig:
        """Get train config from train config file.

        Returns:
            TrainConfig: train config
        """
        if not self.train_config_file.exists():
            warnings.warn("Train config file is not exist. Train first!")
            return None
        return TrainConfig.load(self.train_config_file)

    def get_model_config(self) -> ModelConfig:
        """Get model config from model config file.

        Returns:
            ModelConfig: model config
        """
        return ModelConfig.load(self.model_config_file)

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
        if not self.metric_file.exists():
            warnings.warn("Metric file is not exist. Train first!")
            return []
        return io.load_json(self.metric_file)

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

    # Train Hook
    def before_train(self, cfg: TrainConfig):
        pass

    def on_train_start(self, cfg: TrainConfig):
        pass

    def save_train_config(self, cfg: TrainConfig):
        cfg.save_yaml(self.train_config_file)

    def training(self, cfg: TrainConfig):
        pass

    def on_train_end(self, cfg: TrainConfig):
        pass

    def after_train(self, cfg: TrainConfig, result: TrainResult):
        result.best_ckpt_file = self.best_ckpt_file
        result.last_ckpt_file = self.last_ckpt_file
        result.metrics = self.get_metrics()

    def train(
        self,
        dataset_path: str,
        epochs: int = None,
        batch_size: int = None,
        image_size: Union[int, list[int]] = None,
        learning_rate: float = None,
        letter_box: bool = None,
        pretrained_model: str = None,
        device: str = "0",
        workers: int = 2,
        seed: int = 0,
        verbose: bool = True,
        hold: bool = True,
    ) -> TrainResult:
        """Start Train

        Args:
            dataset_path (str): dataset path
            epochs (int, optional): number of epochs. None to use default. Defaults to None.
            batch_size (int, optional): batch size. None to use default. Defaults to None.
            image_size (Union[int, list[int]], optional): image size. None to use default. Defaults to None.
            learning_rate (float, optional): learning rate. None to use default. Defaults to None.
            letter_box (bool, optional): letter box. None to use default. Defaults to None.
            pretrained_model (str, optional): pretrained model. None to use default. Defaults to None.
            device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
            workers (int, optional): number of workers. Defaults to 2.
            seed (int, optional): random seed. Defaults to 0.
            verbose (bool, optional): verbose. Defaults to True.
            hold (bool, optional): hold process. Defaults to True.

        Raises:
            FileExistsError: if trained artifact exists.
            FileNotFoundError: if can not detect appropriate dataset.
            ValueError: if can not detect appropriate dataset.
            e: something gone wrong with ultralytics

        Example:
            >>> train_result = hub.train(
                    dataset_path=dataset_path,
                    epochs=100,
                    batch_size=16,
                    image_size=640,
                    learning_rate=0.001,
                    letterbox=False,
                    device="0",
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
        
        if self.artifact_dir.exists():
            raise FileExistsError(
                f"{self.artifact_dir}\n"
                "Train artifacts already exist. Remove artifact to re-train (hub.delete_artifact())."
            )

        def inner(callback: TrainCallback, result: TrainResult):
            try:
                self.before_train(cfg)
                self.on_train_start(cfg)
                self.save_train_config(cfg)
                self.training(cfg, callback)
                self.on_train_end(cfg)
                self.after_train(cfg, result)
                callback.force_finish()
            except Exception as e:
                if self.artifact_dir.exists():
                    io.remove_directory(self.artifact_dir)
                callback.force_finish()
                callback.set_failed()
                raise e

        cfg = TrainConfig(
            dataset_path=dataset_path,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=learning_rate,
            letter_box=letter_box,
            pretrained_model=pretrained_model,
            device=device,
            workers=workers,
            seed=seed,
            verbose=verbose,
        )

        callback = TrainCallback(cfg.epochs, self.get_metrics)
        result = TrainResult()
        result.callback = callback

        if hold:
            inner(callback, result)
        else:
            thread = threading.Thread(target=inner, args=(callback, result), daemon=True)
            callback.register_thread(thread)
            callback.start()

        return result

    # Evaluation Hook
    def get_model(self):
        raise NotImplementedError

    def before_evaluate(self, cfg: EvaluateConfig):
        # overwrite training config
        train_config = self.get_train_config()
        if cfg.image_size is None:
            cfg.image_size = train_config.image_size
        if cfg.letter_box is None:
            cfg.letter_box = train_config.letter_box

    def on_evaluate_start(self, cfg: EvaluateConfig):
        pass

    def evaluating(self, cfg: EvaluateConfig, callback: EvaluateCallback) -> str:
        device = cfg.device

        model = self.get_model().to(device)

        dataset = Dataset.load(cfg.dataset_name, cfg.dataset_root_dir)
        dataloader = LabeledDataset(
            dataset,
            cfg.image_size,
            letter_box=cfg.letter_box,
            set_name=cfg.set_name,
        ).get_dataloader(cfg.batch_size, cfg.workers)

        result_parser = get_parser(self.task)(**cfg.to_dict())

        callback._total_steps = len(dataloader)

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
        io.save_json(
            [
                {
                    "tag": tag,
                    "value": value,
                }
                for tag, value in metrics.to_dict().items()
            ],
            self.evaluate_file,
        )

    def on_evaluate_end(self, cfg: EvaluateConfig):
        pass

    def after_evaluate(self, cfg: EvaluateConfig, result: EvaluateResult):
        result.metrics = self.get_evaluate_result()

    def evaluate(
        self,
        dataset_name: str,
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
        dataset_root_dir: str = None,
        hold: bool = True,
    ) -> EvaluateResult:
        """Start Evaluate

        Args:
            dataset_name (str): waffle dataset name.
            batch_size (int, optional): batch size. Defaults to 4.
            image_size (Union[int, list[int]], optional): image size. Defaults to None.
            letter_box (bool, optional): letter box. Defaults to None.
            confidence_threshold (float, optional): confidence threshold. Defaults to 0.25.
            iou_threshold (float, optional): iou threshold. Defaults to 0.5.
            half (bool, optional): half. Defaults to False.
            workers (int, optional): workers. Defaults to 2.
            device (str, optional): device. Defaults to "0".
            draw (bool, optional): draw. Defaults to False.
            dataset_root_dir (str, optional): dataset root dir. Defaults to None.
            hold (bool, optional): hold. Defaults to True.

        Raises:
            FileNotFoundError: if can not detect appropriate dataset.
            e: something gone wrong with ultralytics

        Examples:
            >>> evaluate_result = hub.evaluate(
                    dataset_name="detection_dataset",
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

        def inner(callback: EvaluateCallback, result: EvaluateResult):
            try:
                self.before_evaluate(cfg)
                self.on_evaluate_start(cfg)
                self.evaluating(cfg, callback)
                self.on_evaluate_end(cfg)
                self.after_evaluate(cfg, result)
                callback.force_finish()
            except Exception as e:
                if self.evaluate_file.exists():
                    io.remove_file(self.evaluate_file)
                callback.force_finish()
                callback.set_failed()
                raise e

        cfg = EvaluateConfig(
            dataset_name=dataset_name,
            set_name=set_name,
            batch_size=batch_size,
            image_size=image_size,
            letter_box=letter_box,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            half=half,
            workers=workers,
            device="cpu" if device == "cpu" else f"cuda:{device}",
            draw=draw,
            dataset_root_dir=dataset_root_dir,
        )

        callback = EvaluateCallback(0)
        result = EvaluateResult()
        result.callback = callback

        if hold:
            inner(callback, result)
        else:
            thread = threading.Thread(target=inner, args=(callback, result), daemon=True)
            callback.register_thread(thread)
            callback.start()

        return result

    # inference hooks
    def before_inference(self, cfg: InferenceConfig):
        # overwrite training config
        train_config = self.get_train_config()
        if cfg.image_size is None:
            cfg.image_size = train_config.image_size
        if cfg.letter_box is None:
            cfg.letter_box = train_config.letter_box

    def on_inference_start(self, cfg: InferenceConfig):
        pass

    def inferencing(self, cfg: InferenceConfig, callback: InferenceCallback) -> str:
        device = cfg.device

        model = self.get_model().to(device)
        dataloader = ImageDataset(
            cfg.source, cfg.image_size, letter_box=cfg.letter_box
        ).get_dataloader(cfg.batch_size, cfg.workers)

        result_parser = get_parser(self.task)(**cfg.to_dict())

        results = []
        callback._total_steps = len(dataloader)
        for i, (images, image_infos) in tqdm.tqdm(
            enumerate(dataloader, start=1), total=len(dataloader)
        ):
            result_batch = model(images.to(device))
            result_batch = result_parser(result_batch, image_infos)
            for result, image_info in zip(result_batch, image_infos):
                image_path = image_info.image_path

                relpath = Path(image_path).relative_to(cfg.source)
                results.append({str(relpath): [res.to_dict() for res in result]})

                if cfg.draw:
                    draw = draw_results(
                        image_path,
                        result,
                        names=[x["name"] for x in self.categories],
                    )
                    draw_path = self.draw_dir / relpath.with_suffix(".png")
                    io.make_directory(draw_path.parent)
                    cv2.imwrite(str(draw_path), draw)

            callback.update(i)

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
        source: str,
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
        hold: bool = True,
    ) -> InferenceResult:
        """Start Inference

        Args:
            source (str): source directory
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

        cfg = InferenceConfig(
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

        callback = InferenceCallback(0)
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
    def before_export(self, cfg: ExportConfig):

        # overwrite training config
        train_config = self.get_train_config()
        if cfg.image_size is None:
            cfg.image_size = train_config.image_size

    def on_export_start(self, cfg: ExportConfig):
        pass

    def exporting(self, cfg: ExportConfig, callback: ExportCallback) -> str:
        image_size = cfg.image_size
        image_size = [image_size, image_size] if isinstance(image_size, int) else image_size

        model = self.get_model()

        input_name = ["inputs"]
        if self.task == "object_detection":
            output_names = ["bbox", "conf", "class_id"]
        elif self.task == "classification":
            output_names = ["predictions"]
        else:
            raise NotImplementedError(f"{self.task} does not support export yet.")

        dummy_input = torch.randn(cfg.batch_size, 3, *image_size)

        torch.onnx.export(
            model,
            dummy_input,
            str(self.onnx_file),
            input_names=input_name,
            output_names=output_names,
            opset_version=cfg.opset_version,
            dynamic_axes={name: {0: "batch_size"} for name in input_name + output_names},
        )

    def on_export_end(self, cfg: ExportConfig):
        pass

    def after_export(self, cfg: ExportConfig, result: ExportResult):
        result.export_file = self.onnx_file

    def export(
        self,
        image_size: Union[int, list[int]] = None,
        batch_size: int = 16,
        opset_version: int = 11,
        hold: bool = True,
    ) -> ExportResult:
        """Export Model

        Args:
            image_size (Union[int, list[int]], optional): inference image size. None for same with train_config (recommended).
            batch_size (int, optional): dynamic batch size. Defaults to 16.
            opset_version (int, optional): onnx opset version. Defaults to 11.
            hold (bool, optional): hold or not.
                If True then it holds until task finished.
                If False then return Inferece Callback and run in background. Defaults to True.

        Example:
            >>> export_result = hub.export(
                image_size=640,
                batch_size=16,
                opset_version=11,
            )
            # or simply use train option by passing None
            >>> export_result = hub.export(
                ...,
                image_size=None,  # use train option
                ...
            )
            >>> export_result.export_file
            hubs/my_hub/weights/model.onnx

        Returns:
            ExportResult: export result
        """
        self.check_train_sanity()

        def inner(callback: ExportCallback, result: ExportResult):
            try:
                self.before_export(cfg)
                self.on_export_start(cfg)
                self.exporting(cfg, callback)
                self.on_export_end(cfg)
                self.after_export(cfg, result)
                callback.force_finish()
            except Exception as e:
                if self.onnx_file.exists():
                    io.remove_file(self.onnx_file)
                callback.force_finish()
                callback.set_failed()
                raise e

        cfg = ExportConfig(
            image_size=image_size,
            batch_size=batch_size,
            opset_version=opset_version,
        )

        callback = ExportCallback(1)
        result = ExportResult()
        result.callback = callback

        if hold:
            inner(callback, result)
        else:
            thread = threading.Thread(target=inner, args=(callback, result), daemon=True)
            callback.register_thread(thread)
            callback.start()

        return result
