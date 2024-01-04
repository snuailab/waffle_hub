from abc import ABC

from waffle_utils.callback import BaseCallback

from waffle_hub.hub.onnx.exporter import OnnxExporter


class BaseExportOnnxCallback(BaseCallback, ABC):
    def __init__(self):
        pass

    def setup(self, onnx_exporter: OnnxExporter) -> None:
        """Called when worker starts."""

    def teardown(self, onnx_exporter: OnnxExporter) -> None:
        """Called when worker ends."""

    def before_export_onnx(self, onnx_exporter: OnnxExporter) -> None:
        """Called when the export_onnx begins."""

    def on_export_onnx_start(self, onnx_exporter: OnnxExporter) -> None:
        """Called when the export_onnx function begins."""

    def on_export_onnx_end(self, onnx_exporter: OnnxExporter) -> None:
        """Called when the export_onnx loop step ends."""

    def after_export_onnx(self, onnx_exporter: OnnxExporter) -> None:
        """Called when the export_onnx ends."""

    def on_exception_stopped(self, onnx_exporter: OnnxExporter, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""

    def on_exception_failed(self, onnx_exporter: OnnxExporter, e: Exception) -> None:
        """Called when an error occurs"""
