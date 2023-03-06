from pathlib import Path

import pytest
from waffle_utils.dataset import Dataset
from waffle_utils.file import io, network

from waffle_hub.hub import UltralyticsHub


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
    return ds


def test_ultralytics_detection_train(tmpdir: Path, dummy_dataset: Dataset):

    dummy_dataset.split_train_val(0.8)
    export_dir = dummy_dataset.export("yolo_detection")
    hub = UltralyticsHub(
        name="test_det",
        task="detect",
        model_name="yolov8",
        model_size="n",
        pretrained_model=None,
        root_dir=tmpdir,
    )
    hub.train(
        dataset_dir=export_dir,
        epochs=1,
        batch_size=4,
        image_size=64,
        device="0",
    )


def test_ultralytics_detect_train(tmpdir: Path, dummy_dataset: Dataset):

    dummy_dataset.split_train_val(0.8)
    export_dir = dummy_dataset.export("yolo_detection")
    hub = UltralyticsHub(
        name="test_det",
        task="detect",
        model_name="yolov8",
        model_size="n",
        pretrained_model=None,
        root_dir=tmpdir,
    )
    hub.train(
        dataset_dir=export_dir,
        epochs=1,
        batch_size=4,
        image_size=64,
        device="0",
    )
    assert hub.check_train_sanity()


def test_ultralytics_classify_train(tmpdir: Path, dummy_dataset: Dataset):

    dummy_dataset.split_train_val(0.8)
    export_dir = dummy_dataset.export("yolo_classification")
    hub = UltralyticsHub(
        name="test_cls",
        task="classify",
        model_name="yolov8",
        model_size="n",
        pretrained_model=None,
        root_dir=tmpdir,
    )
    hub.train(
        dataset_dir=export_dir,
        epochs=1,
        batch_size=4,
        image_size=28,
        device="0",
    )
    assert hub.check_train_sanity()
