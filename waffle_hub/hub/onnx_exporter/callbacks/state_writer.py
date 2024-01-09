from pathlib import Path
from typing import Union

from waffle_hub.hub.onnx_exporter.callbacks import BaseExportOnnxCallback
from waffle_hub.hub.onnx_exporter.exporter import OnnxExporter


class ExportOnnxStateWriterCallback(BaseExportOnnxCallback):
    def __init__(self, save_path: Union[str, Path]):
        self.save_path = Path(save_path)

    def setup(self, onnx_exporter: OnnxExporter) -> None:
        """Called when worker starts."""
        self._save_state(onnx_exporter.state)

    def teardown(self, onnx_exporter: OnnxExporter) -> None:
        """Called when worker ends."""

    def before_export_onnx(self, onnx_exporter: OnnxExporter) -> None:
        """Called when the export_onnx begins."""
        self._save_state(onnx_exporter.state)

    def on_export_onnx_start(self, onnx_exporter: OnnxExporter) -> None:
        """Called when the export_onnx function begins."""
        self._save_state(onnx_exporter.state)

    def on_export_onnx_end(self, onnx_exporter: OnnxExporter) -> None:
        """Called when the export_onnx loop step ends."""
        self._save_state(onnx_exporter.state)

    def after_export_onnx(self, onnx_exporter: OnnxExporter) -> None:
        """Called when the export_onnx ends."""
        self._save_state(onnx_exporter.state)

    def on_exception_stopped(self, onnx_exporter: OnnxExporter, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""
        self._save_state(onnx_exporter.state)

    def on_exception_failed(self, onnx_exporter: OnnxExporter, e: Exception) -> None:
        """Called when an error occurs"""
        self._save_state(onnx_exporter.state)

    def _save_state(self, state):
        state.save_json(save_path=self.save_path)
