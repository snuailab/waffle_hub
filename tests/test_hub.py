from pathlib import Path

import pytest
from waffle_utils.dataset import Dataset
from waffle_utils.file import io, network

from waffle_hub.hub.adapter.ultralytics import UltralyticsHub


@pytest.fixture
def dummy_dataset(tmpdir: Path):
    url = "https://github.com/snuailab/waffle_utils/raw/main/mnist.zip"

    dummy_zip_file = tmpdir / "mnist.zip"
    dummy_extract_dir = tmpdir / "extract"
    dummy_coco_root_dir = tmpdir / "extract/raw"
    dummy_coco_file = tmpdir / "extract/exports/coco.json"

    network.get_file_from_url(url, dummy_zip_file, create_directory=True)
    io.unzip(dummy_zip_file, dummy_extract_dir, create_directory=True)

    ds = Dataset.from_coco(
        "mnist", dummy_coco_file, Path(dummy_coco_root_dir), root_dir=tmpdir
    )
    ds.split(0.8)
    return ds


def test_ultralytics_detect_train_inference(
    tmpdir: Path, dummy_dataset: Dataset
):

    export_dir = dummy_dataset.export("yolo_detection")

    name = "test_det"

    hub = UltralyticsHub(
        name=name,
        task="object_detection",
        model_type="yolov8",
        model_size="n",
        classes=["1", "2"],
        root_dir=tmpdir,
    )
    hub = UltralyticsHub.load(name=name, root_dir=tmpdir)
    hub: UltralyticsHub = UltralyticsHub.from_model_config(
        name=name,
        model_config_file=tmpdir / name / UltralyticsHub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )
    hub.train(
        dataset_path=export_dir,
        epochs=1,
        batch_size=4,
        image_size=32,
        pretrained_model=None,
        device="cpu",
    )
    assert hub.check_train_sanity()

    inference_dir = hub.inference(
        source=export_dir,
        device="cpu",
    )
    assert (Path(inference_dir) / "results").exists()

    onnx_file = hub.export()
    assert Path(onnx_file).exists()


def test_ultralytics_classify_train(tmpdir: Path, dummy_dataset: Dataset):

    name = "test_cls"

    export_dir = dummy_dataset.export("yolo_classification")
    hub = UltralyticsHub(
        name=name,
        task="classification",
        model_type="yolov8",
        model_size="n",
        classes=["1", "2"],
        root_dir=tmpdir,
    )
    hub.train(
        dataset_path=export_dir,
        epochs=1,
        batch_size=4,
        image_size=28,
        pretrained_model=None,
        device="cpu",
    )
    assert hub.check_train_sanity()

    inference_dir = hub.inference(
        source=export_dir,
        device="cpu",
    )
    assert (Path(inference_dir) / "results").exists()

    onnx_file = hub.export()
    assert Path(onnx_file).exists()
