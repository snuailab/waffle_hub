from typing import Any

from waffle_utils.callback import BaseCallback
from waffle_utils.hook import BaseHook

from waffle_hub import ExportOnnxStatus
from waffle_hub.utils.process import _register_signal_handler


class BaseExportOnnxHook(BaseHook):
    def __init__(self, callbacks: list[BaseCallback] = None):
        super().__init__(callbacks)

    def setup(self) -> None:
        """Called when worker starts."""
        _register_signal_handler()

    def teardown(self) -> None:
        """Called when worker ends."""

    def before_export_onnx(self) -> None:
        """Called when the export_onnx begins."""

    def on_export_onnx_start(self) -> None:
        """Called when the export_onnx function begins."""
        self.state.status = ExportOnnxStatus.RUNNING
        self.state.clear_error()

    def on_export_onnx_end(self) -> None:
        """Called when the export_onnx loop step ends."""
        self.result.onnx_file = self.onnx_file

    def after_export_onnx(self) -> None:
        """Called when the export_onnx ends."""
        self.state.status = ExportOnnxStatus.SUCCESS

    def on_exception_stopped(self, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""
        self.state.status = ExportOnnxStatus.STOPPED
        self.state.set_error(e)

    def on_exception_failed(self, e: Exception) -> None:
        """Called when an error occurs"""
        self.state.status = ExportOnnxStatus.FAILED
        self.state.set_error(e)
