import threading
import warnings
from pathlib import Path
from typing import Union

import torch
from torch import nn
from waffle_utils.file import io
from waffle_utils.utils import type_validator

from waffle_dough.type.task_type import TaskType
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import ExportOnnxConfig, TrainConfig
from waffle_hub.schema.result import ExportOnnxResult
from waffle_hub.utils.callback import ExportCallback
from waffle_hub.utils.memory import device_context


class OnnxExporter:
    """
    export onnx manager class
    """

    # directory settting
    EXPORT_DIR = Path("weights")

    # export results file path ##--
    ONNX_FILE = EXPORT_DIR / "model.onnx"

    def __init__(
        self,
        root_dir: Path,
        model: Union[ModelWrapper, nn.Module],
        task: Union[str, TaskType],
        train_config: TrainConfig = None,
    ):
        self.root_dir = Path(root_dir)
        self.model = model
        self.task = task
        self.train_config = train_config

    # properties
    @property
    def task(self) -> str:
        """Task Name"""
        return self.__task

    @task.setter
    def task(self, v):
        if v not in list(TaskType):
            raise ValueError(f"Invalid task type: {v}" f"Available task types: {list(TaskType)}")
        self.__task = str(v.value) if isinstance(v, TaskType) else str(v).lower()

    @property
    def export_dir(self) -> Path:
        """Export Directory"""
        return self.root_dir / self.EXPORT_DIR

    @property
    def onnx_file(self) -> Path:
        """Best Checkpoint ONNX File"""
        return self.root_dir / self.ONNX_FILE

    # methods
    def export(
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
            >>> exporter = OnnxExporter(...)
            >>> export_onnx_result = exporter.export(
                image_size=640,
                batch_size=16,
                opset_version=11,
            )
            # or simply use train option by passing None
            >>> exporter = OnnxExporter(..., train_config=train_config)
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

        @device_context("cpu" if device == "cpu" else device)
        def inner(callback: ExportCallback, result: ExportOnnxResult):
            try:
                self.before_export()
                self.on_export_start()
                self.exporting()
                self.on_export_end()
                self.after_export(result)
                callback.force_finish()
            except Exception as e:
                if self.onnx_file.exists():
                    io.remove_file(self.onnx_file)
                callback.force_finish()
                callback.set_failed()
                raise e

        # overwrite training config or default
        if image_size is None:
            if self.train_config is not None:
                image_size = self.train_config.image_size
            else:
                image_size = 224  # default image size

        self.cfg = ExportOnnxConfig(
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

    # Export Hook
    def before_export(self):
        pass

    def on_export_start(self):
        pass

    def exporting(self):
        image_size = self.cfg.image_size
        image_size = [image_size, image_size] if isinstance(image_size, int) else image_size

        model = self.model.half() if self.cfg.half else self.model
        model = model.to(self.cfg.device)

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
            self.cfg.batch_size,
            3,
            *image_size,
            dtype=torch.float16 if self.cfg.half else torch.float32,
        )
        dummy_input = dummy_input.to(self.cfg.device)

        torch.onnx.export(
            model,
            dummy_input,
            str(self.onnx_file),
            input_names=input_name,
            output_names=output_names,
            opset_version=self.cfg.opset_version,
            dynamic_axes={name: {0: "batch_size"} for name in input_name + output_names},
        )

    def on_export_end(self):
        pass

    def after_export(self, result: ExportOnnxResult):
        result.onnx_file = self.onnx_file
