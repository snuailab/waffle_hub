import time
from pathlib import Path

import pytest
from waffle_utils.dataset import Dataset
from waffle_utils.file import io, network

from waffle_hub.hub.adapter.ultralytics import UltralyticsHub
from waffle_hub.utils.callback import (
    ExportCallback,
    InferenceCallback,
    TrainCallback,
)


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


def test_ultralytics_object_detection(tmpdir: Path, dummy_dataset: Dataset):

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
    train_callback: TrainCallback = hub.train(
        dataset_path=export_dir,
        epochs=1,
        batch_size=4,
        image_size=32,
        pretrained_model=None,
        device="cpu",
    )
    assert train_callback.get_progress() == 1
    assert len(train_callback.get_metrics()) == 1
    assert Path(train_callback.best_ckpt_file).exists()
    assert Path(train_callback.last_ckpt_file).exists()
    assert Path(train_callback.metric_file).exists()
    assert Path(train_callback.result_dir).exists()

    inference_callback: InferenceCallback = hub.inference(
        source=export_dir,
        device="cpu",
    )
    assert inference_callback.get_progress() == 1
    assert Path(inference_callback.inference_dir).exists()

    export_callback: ExportCallback = hub.export()
    assert Path(export_callback.export_file).exists()


def test_ultralytics_classification(tmpdir: Path, dummy_dataset: Dataset):

    export_dir = dummy_dataset.export("yolo_classification")

    name = "test_cls"

    hub = UltralyticsHub(
        name=name,
        task="classification",
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
    train_callback: TrainCallback = hub.train(
        dataset_path=export_dir,
        epochs=1,
        batch_size=4,
        image_size=32,
        pretrained_model=None,
        device="cpu",
    )
    assert train_callback.get_progress() == 1
    assert len(train_callback.get_metrics()) == 1
    assert Path(train_callback.best_ckpt_file).exists()
    assert Path(train_callback.last_ckpt_file).exists()
    assert Path(train_callback.metric_file).exists()
    assert Path(train_callback.result_dir).exists()

    inference_callback: InferenceCallback = hub.inference(
        source=export_dir,
        device="cpu",
    )
    assert inference_callback.get_progress() == 1
    assert Path(inference_callback.inference_dir).exists()

    export_callback: ExportCallback = hub.export()
    assert Path(export_callback.export_file).exists()


def test_non_hold(tmpdir: Path, dummy_dataset: Dataset):

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
    train_callback: TrainCallback = hub.train(
        dataset_path=export_dir,
        epochs=1,
        batch_size=4,
        image_size=32,
        pretrained_model=None,
        device="cpu",
        hold=False,
    )

    while not train_callback.is_finished():
        time.sleep(0.1)

    assert train_callback.get_progress() == 1
    assert len(train_callback.get_metrics()) == 1
    assert Path(train_callback.best_ckpt_file).exists()
    assert Path(train_callback.last_ckpt_file).exists()
    assert Path(train_callback.metric_file).exists()
    assert Path(train_callback.result_dir).exists()

    inference_callback: InferenceCallback = hub.inference(
        source=export_dir, device="cpu", hold=False
    )
    while not inference_callback.is_finished():
        time.sleep(0.1)

    assert inference_callback.get_progress() == 1
    assert Path(inference_callback.inference_dir).exists()

    export_callback: ExportCallback = hub.export(hold=False)

    while not export_callback.is_finished():
        time.sleep(0.1)

    assert Path(export_callback.export_file).exists()
