import warnings
from pathlib import Path
from typing import Union

import torch
from waffle_utils.callback import BaseCallback
from waffle_utils.file import io

from waffle_hub import ExportOnnxStatus
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import ExportOnnxConfig
from waffle_hub.schema.result import ExportOnnxResult
from waffle_hub.schema.state import ExportOnnxState
from waffle_hub.type import TaskType
from waffle_hub.utils.memory import device_context

from .hook import BaseExportOnnxHook


class OnnxExporter(BaseExportOnnxHook):
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
        model: ModelWrapper,
        callbacks: list[BaseCallback] = None,
    ):
        super().__init__(callbacks)
        self.root_dir = Path(root_dir)
        self.model = model
        self.state = ExportOnnxState(status=ExportOnnxStatus.INIT)
        self.result = ExportOnnxResult()

    # properties
    @property
    def export_dir(self) -> Path:
        """Export Directory"""
        return self.root_dir / self.EXPORT_DIR

    @property
    def onnx_file(self) -> Path:
        """Best Checkpoint ONNX File"""
        return self.root_dir / self.ONNX_FILE

    # methods
    @device_context
    def export(
        self,
        image_size: Union[int, list[int]] = [640, 640],
        batch_size: int = 16,
        opset_version: int = 11,
        half: bool = False,
        device: str = "0",
    ) -> ExportOnnxResult:
        """Export Onnx Model

        Args:
            image_size (Union[int, list[int]], optional): Export image size. Default to [640, 640].
            batch_size (int, optional): dynamic batch size. Defaults to 16.
            opset_version (int, optional): onnx opset version. Defaults to 11.
            half (bool, optional): half. Defaults to False.
            device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".

        Example:
            >>> exporter = OnnxExporter(...)
            >>> export_onnx_result = exporter.export(
                image_size=640,
                batch_size=16,
                opset_version=11,
            )
            >>> export_onnx_result.onnx_file
            hubs/my_hub/weights/model.onnx

        Returns:
            ExportOnnxResult: export onnx result
        """

        try:
            self.run_default_hook("setup")
            self.run_callback_hooks("setup", self)

            self.cfg = ExportOnnxConfig(
                image_size=image_size if isinstance(image_size, list) else [image_size, image_size],
                batch_size=batch_size,
                opset_version=opset_version,
                half=half,
                device="cpu" if device == "cpu" else f"cuda:{device}",
            )
            self.run_default_hook("before_export_onnx")
            self.run_callback_hooks("before_export_onnx", self)

            self._export_onnx()

            self.run_default_hook("after_export_onnx")
            self.run_callback_hooks("after_export_onnx", self)

        except (KeyboardInterrupt, SystemExit) as e:
            self.run_default_hook("on_exception_stopped", e)
            self.run_callback_hooks("on_exception_stopped", self, e)
            raise e
        except Exception as e:
            self.run_default_hook("on_exception_failed", e)
            self.run_callback_hooks("on_exception_failed", self, e)
            if self.onnx_file.exists():
                io.remove_file(self.onnx_file)
            raise e
        finally:
            self.run_default_hook("teardown")
            self.run_callback_hooks("teardown", self)

        return self.result

    def _export_onnx(self):
        self.run_default_hook("on_export_onnx_start")
        self.run_callback_hooks("on_export_onnx_start", self)

        image_size = self.cfg.image_size
        image_size = [image_size, image_size] if isinstance(image_size, int) else image_size

        model = self.model.half() if self.cfg.half else self.model
        model = model.to(self.cfg.device)

        input_name = ["inputs"]
        if self.model.task == TaskType.OBJECT_DETECTION:
            output_names = ["bbox", "conf", "class_id"]
        elif self.model.task == TaskType.CLASSIFICATION:
            output_names = ["predictions"]
        elif self.model.task == TaskType.INSTANCE_SEGMENTATION:
            output_names = ["bbox", "conf", "class_id", "masks"]
        elif self.model.task == TaskType.TEXT_RECOGNITION:
            output_names = ["class_ids", "confs"]
        elif self.model.task == TaskType.SEMANTIC_SEGMENTATION:
            output_names = ["score_map"]
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

        self.run_default_hook("on_export_onnx_end")
        self.run_callback_hooks("on_export_onnx_end", self)
