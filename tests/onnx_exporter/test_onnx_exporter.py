from pathlib import Path

from waffle_hub import ExportOnnxStatus
from waffle_hub.hub.model.boring_wrapper import BoringModelWrapper
from waffle_hub.hub.onnx_exporter.exporter import OnnxExporter
from waffle_hub.type import TaskType


def test_onnx_exporter(tmpdir: Path):

    model = BoringModelWrapper(
        task=TaskType.OBJECT_DETECTION, categories=["1", "2", "3"], batch_size=1
    )

    onnx_exporter = OnnxExporter(root_dir=tmpdir, model=model)

    assert onnx_exporter.state.status == ExportOnnxStatus.INIT

    result = onnx_exporter.export(
        image_size=[32, 32],
        device="cpu",
    )

    assert onnx_exporter.state.status == ExportOnnxStatus.SUCCESS
    assert onnx_exporter.result == result
    assert onnx_exporter.onnx_file.exists()
