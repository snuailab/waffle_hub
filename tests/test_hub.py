import time
from pathlib import Path

import pytest
import torch

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub.adapter.autocare_dlt import AutocareDLTHub
from waffle_hub.hub.adapter.hugging_face import HuggingFaceHub
from waffle_hub.hub.adapter.ultralytics import UltralyticsHub
from waffle_hub.schema.result import (
    EvaluateResult,
    ExportResult,
    InferenceResult,
    TrainResult,
)


@pytest.fixture
def instance_segmentation_dataset(coco_path: Path, tmpdir: Path):
    dataset: Dataset = Dataset.from_coco(
        name="seg",
        task=TaskType.INSTANCE_SEGMENTATION,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    dataset.split(0.2, 0.2, 0.6)

    return dataset


@pytest.fixture
def object_detection_dataset(coco_path: Path, tmpdir: Path):
    dataset: Dataset = Dataset.from_coco(
        name="od",
        task=TaskType.OBJECT_DETECTION,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    dataset.split(0.2, 0.2, 0.6)

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
    dataset.split(0.2, 0.2, 0.6)

    return dataset


def _train(hub, dataset: Dataset, image_size: int, hold: bool = True):
    result: TrainResult = hub.train(
        dataset_path=dataset.export(hub.backend),
        epochs=1,
        image_size=image_size,
        batch_size=4,
        pretrained_model=None,
        device="cpu",
        workers=0,
        hold=hold,
    )

    if not hold:
        assert hasattr(result, "callback")
        while not result.callback.is_finished() and not result.callback.is_failed():
            time.sleep(1)
        assert result.callback.is_finished()
        assert not result.callback.is_failed()

    print(hub.metric_file, result.metrics)
    assert len(result.metrics) >= 1
    assert Path(result.best_ckpt_file).exists()
    # assert Path(result.last_ckpt_file).exists()

    return result


def _evaluate(hub, dataset: Dataset, hold: bool = True):

    result: EvaluateResult = hub.evaluate(
        dataset_name=dataset.name,
        dataset_root_dir=dataset.root_dir,
        device="cpu",
        workers=0,
        hold=hold,
    )

    if not hold:
        assert hasattr(result, "callback")
        while not result.callback.is_finished() and not result.callback.is_failed():
            time.sleep(1)
        assert result.callback.is_finished()
        assert not result.callback.is_failed()

    assert len(result.metrics) >= 1

    return result


def _inference(hub, source: str, hold: bool = True):

    result: InferenceResult = hub.inference(
        source=source,
        draw=True,
        device="cpu",
        workers=0,
        hold=hold,
    )

    if not hold:
        assert hasattr(result, "callback")
        while not result.callback.is_finished() and not result.callback.is_failed():
            time.sleep(1)
        assert result.callback.is_finished()
        assert not result.callback.is_failed()

    assert len(result.predictions) >= 1
    assert Path(result.draw_dir).exists()

    return result


def _export(hub, half: bool = False, hold: bool = True):
    result: ExportResult = hub.export(
        hold=hold,
        half=half,
        device="cpu",
    )

    if not hold:
        assert hasattr(result, "callback")
        while not result.callback.is_finished() and not result.callback.is_failed():
            time.sleep(1)
        assert result.callback.is_finished()
        assert not result.callback.is_failed()

    assert Path(result.export_file).exists()

    return result


def _feature_extraction(
    hub,
    image_size: int,
):

    model = hub.get_model()

    layer_names = model.get_layer_names()
    assert len(layer_names) > 0

    x = torch.randn(1, 3, image_size, image_size)
    layer_name = layer_names[-1]
    x, feature_maps = model.get_feature_maps(x, layer_name)
    assert len(feature_maps) == 1


def _benchmark(hub, image_size):
    hub.benchmark(device="cpu", half=False, image_size=image_size)


def _total(hub, dataset: Dataset, image_size: int, hold: bool = True):

    _train(hub, dataset, image_size, hold=hold)
    _evaluate(hub, dataset, hold=hold)
    _inference(hub, dataset.raw_image_dir, hold=hold)
    _export(hub, half=False, hold=hold)
    # _export(hub, half=True, hold=hold)  # cpu cannot be half
    _feature_extraction(hub, image_size)
    _benchmark(hub, image_size)


def test_ultralytics_segmentation(instance_segmentation_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = instance_segmentation_dataset

    # test hub
    name = "test_seg"
    hub = UltralyticsHub.new(
        name=name,
        task=TaskType.INSTANCE_SEGMENTATION,
        model_type="yolov8",
        model_size="n",
        categories=dataset.category_names,
        root_dir=tmpdir,
    )
    hub = UltralyticsHub.load(name=name, root_dir=tmpdir)
    hub: UltralyticsHub = UltralyticsHub.from_model_config(
        name=name,
        model_config_file=tmpdir / name / UltralyticsHub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_ultralytics_object_detection(object_detection_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = object_detection_dataset

    # test hub
    name = "test_det"
    hub = UltralyticsHub.new(
        name=name,
        task=TaskType.OBJECT_DETECTION,
        model_type="yolov8",
        model_size="n",
        categories=dataset.category_names,
        root_dir=tmpdir,
    )
    hub = UltralyticsHub.load(name=name, root_dir=tmpdir)
    hub: UltralyticsHub = UltralyticsHub.from_model_config(
        name=name,
        model_config_file=tmpdir / name / UltralyticsHub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_ultralytics_classification(classification_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = classification_dataset

    # test hub
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

    _total(hub, dataset, image_size)


def test_huggingface_object_detection(object_detection_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = object_detection_dataset

    # test hub
    name = "test_det"
    hub = HuggingFaceHub.new(
        name=name,
        task=TaskType.OBJECT_DETECTION,
        model_type="YOLOS",
        model_size="tiny",
        categories=object_detection_dataset.category_names,
        root_dir=tmpdir,
    )
    hub = HuggingFaceHub.load(name=name, root_dir=tmpdir)
    hub: HuggingFaceHub = HuggingFaceHub.from_model_config(
        name=name,
        model_config_file=tmpdir / name / HuggingFaceHub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_huggingface_classification(classification_dataset: Dataset, tmpdir: Path):
    image_size = 224
    dataset = classification_dataset

    # test hub
    name = "test_cls"
    hub = HuggingFaceHub.new(
        name=name,
        task=TaskType.CLASSIFICATION,
        model_type="ViT",
        model_size="tiny",
        categories=classification_dataset.category_names,
        root_dir=tmpdir,
    )
    hub = HuggingFaceHub.load(name=name, root_dir=tmpdir)
    hub: HuggingFaceHub = HuggingFaceHub.from_model_config(
        name=name,
        model_config_file=tmpdir / name / HuggingFaceHub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_non_hold(classification_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = classification_dataset

    # test hub
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

    _total(hub, dataset, image_size, hold=False)


def test_autocare_dlt_object_detection(object_detection_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = object_detection_dataset

    # test hub
    name = "test_det"
    hub = AutocareDLTHub.new(
        name=name,
        task=TaskType.OBJECT_DETECTION,
        model_type="YOLOv5",
        model_size="s",
        categories=object_detection_dataset.category_names,
        root_dir=tmpdir,
    )
    hub = AutocareDLTHub.load(name=name, root_dir=tmpdir)
    hub: AutocareDLTHub = AutocareDLTHub.from_model_config(
        name=name,
        model_config_file=tmpdir / name / AutocareDLTHub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_autocare_dlt_classification(classification_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = classification_dataset

    # temporal solution
    super_cat = [[c.supercategory, c.name] for c in dataset.categories.values()]
    super_cat_dict = {}
    for super_cat, cat in super_cat:
        if super_cat not in super_cat_dict:
            super_cat_dict[super_cat] = []
        super_cat_dict[super_cat].append(cat)
    super_cat_dict_list = []

    for super_cat, cat in super_cat_dict.items():
        super_cat_dict_list.append({super_cat: cat})

    # test hub
    name = "test_cls"
    hub = AutocareDLTHub.new(
        name=name,
        task=TaskType.CLASSIFICATION,
        model_type="Classifier",
        model_size="s",
        categories=super_cat_dict_list,
        root_dir=tmpdir,
    )
    hub = AutocareDLTHub.load(name=name, root_dir=tmpdir)
    hub: AutocareDLTHub = AutocareDLTHub.from_model_config(
        name=name,
        model_config_file=tmpdir / name / AutocareDLTHub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)
