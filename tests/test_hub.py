import time
from pathlib import Path

import pytest
import torch

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub.adapter.ultralytics import UltralyticsHub
from waffle_hub.utils.callback import (
    ExportCallback,
    InferenceCallback,
    TrainCallback,
)


@pytest.fixture
def object_detection_dataset(coco_path: Path, tmpdir: Path):
    dataset: Dataset = Dataset.from_coco(
        name="od",
        task=TaskType.OBJECT_DETECTION,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    dataset.split(0.8)

    return dataset


@pytest.fixture
def classification_dataset(coco_path: Path, tmpdir: Path):
    dataset: Dataset = Dataset.from_coco(
        name="cls",
        task=TaskType.CLASSIFICATION,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    dataset.split(0.8)

    return dataset


def test_ultralytics_object_detection(
    object_detection_dataset: Dataset, tmpdir: Path
):

    export_dir = object_detection_dataset.export("yolo")

    name = "test_det"
    hub = UltralyticsHub.new(
        name=name,
        task=TaskType.OBJECT_DETECTION,
        model_type="yolov8",
        model_size="n",
        categories=object_detection_dataset.category_names,
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
        draw=True,
        device="cpu",
    )
    assert inference_callback.get_progress() == 1
    assert Path(inference_callback.inference_dir).exists()

    export_callback: ExportCallback = hub.export()
    assert Path(export_callback.export_file).exists()

    model = hub.get_model()

    layer_names = model.get_layer_names()
    assert len(layer_names) > 0

    x = torch.randn(4, 3, 64, 64)
    layer_name = layer_names[-1]
    x, feature_maps = model.get_feature_maps(x, layer_name)
    assert len(feature_maps) == 1


def test_ultralytics_classification(
    classification_dataset: Dataset, tmpdir: Path
):

    export_dir = classification_dataset.export("yolo")

    name = "test_cls"
    hub = UltralyticsHub.new(
        name=name,
        task=TaskType.CLASSIFICATION,
        model_type="yolov8",
        model_size="n",
        categories=classification_dataset.category_names,
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
        draw=True,
        device="cpu",
    )
    assert inference_callback.get_progress() == 1
    assert Path(inference_callback.inference_dir).exists()

    export_callback: ExportCallback = hub.export()
    assert Path(export_callback.export_file).exists()

    model = hub.get_model()

    layer_names = model.get_layer_names()
    assert len(layer_names) > 0

    x = torch.randn(4, 3, 64, 64)
    layer_name = layer_names[-1]
    x, feature_maps = model.get_feature_maps(x, layer_name)
    assert len(feature_maps) == 1


def test_non_hold(classification_dataset: Dataset, tmpdir: Path):

    export_dir = classification_dataset.export("yolo")

    name = "test_hold"

    hub = UltralyticsHub.new(
        name=name,
        task=TaskType.CLASSIFICATION,
        model_type="yolov8",
        model_size="n",
        categories=classification_dataset.category_names,
        root_dir=tmpdir,
    )
    hub = UltralyticsHub.load(name=name, root_dir=tmpdir)
    hub: UltralyticsHub = UltralyticsHub.from_model_config(
        name=name,
        model_config_file=tmpdir / name / UltralyticsHub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    # fail case
    try:
        train_callback: TrainCallback = hub.train(
            dataset_path="dummy no data",
            epochs=1,
            batch_size=4,
            image_size=32,
            pretrained_model=None,
            device="cpu",
            hold=False,
        )
        while not train_callback.is_finished():
            time.sleep(1)
    except Exception:
        pass
    assert train_callback.is_failed()

    # success case
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
        time.sleep(1)
    while not Path(train_callback.best_ckpt_file).exists():
        time.sleep(1)

    assert not train_callback.is_failed()

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
        time.sleep(1)
    while not Path(inference_callback.inference_dir).exists():
        time.sleep(1)

    assert not inference_callback.is_failed()

    assert inference_callback.get_progress() == 1
    assert Path(inference_callback.inference_dir).exists()

    export_callback: ExportCallback = hub.export(hold=False)

    while not export_callback.is_finished():
        time.sleep(1)
    while not Path(export_callback.export_file).exists():
        time.sleep(1)

    assert not export_callback.is_failed()

    assert Path(export_callback.export_file).exists()
