from pathlib import Path

from waffle_utils.file import io

from waffle_hub import ExportOnnxStatus
from waffle_hub.hub.model.boring_wrapper import BoringModelWrapper
from waffle_hub.hub.onnx_exporter.callbacks.state_writer import (
    ExportOnnxStateWriterCallback,
)
from waffle_hub.hub.onnx_exporter.exporter import OnnxExporter
from waffle_hub.type import TaskType


def test_export_state_writer_callback(tmpdir: Path):
    # set up
    state_save_path = tmpdir / "export_onnx_state.json"
    model = BoringModelWrapper(
        task=TaskType.OBJECT_DETECTION, categories=["1", "2", "3"], batch_size=1
    )

    onnx_exporter = OnnxExporter(
        root_dir=tmpdir,
        model=model,
        callbacks=[ExportOnnxStateWriterCallback(save_path=state_save_path)],
    )

    assert onnx_exporter.state.status == ExportOnnxStatus.INIT
    assert not state_save_path.exists()

    result = onnx_exporter.export(
        image_size=[32, 32],
        device="cpu",
    )

    assert onnx_exporter.state.status == ExportOnnxStatus.SUCCESS
    assert onnx_exporter.result == result
    assert onnx_exporter.onnx_file.exists()
    assert state_save_path.exists()

    temp_state = io.load_json(state_save_path)
    assert temp_state["status"] == ExportOnnxStatus.SUCCESS
