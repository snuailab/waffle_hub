import time
from pathlib import Path

import torch

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub
from waffle_hub.schema.result import (
    EvaluateResult,
    ExportResult,
    InferenceResult,
    TrainResult,
)


def _train(hub, dataset: Dataset, image_size: int, hold: bool = True):
    result: TrainResult = hub.train(
        dataset=dataset,
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

    assert len(result.metrics) >= 1
    assert len(result.eval_metrics) >= 1
    assert Path(result.best_ckpt_file).exists()
    # assert Path(result.last_ckpt_file).exists()

    return result


def _evaluate(hub, dataset: Dataset, hold: bool = True):

    result: EvaluateResult = hub.evaluate(
        dataset=dataset,
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

    assert len(result.eval_metrics) >= 1

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


def _util(hub):
    name = hub.name
    backend = hub.backend
    root_dir = hub.root_dir

    hub_class = Hub.get_hub_class(backend)
    assert hub_class == type(hub)

    hub_loaded = Hub.load(name, root_dir)
    assert isinstance(hub_loaded, type(hub))


def _total(hub, dataset: Dataset, image_size: int, hold: bool = True):

    _train(hub, dataset, image_size, hold=hold)
    _evaluate(hub, dataset, hold=hold)
    _inference(hub, dataset.raw_image_dir, hold=hold)
    _export(hub, half=False, hold=hold)
    # _export(hub, half=True, hold=hold)  # cpu cannot be half
    _feature_extraction(hub, image_size)
    _benchmark(hub, image_size)
    _util(hub)


def test_ultralytics_segmentation(instance_segmentation_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = instance_segmentation_dataset

    # test hub
    name = "test_seg"
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.INSTANCE_SEGMENTATION,
        model_type="yolov8",
        model_size="n",
        categories=dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_ultralytics_object_detection(object_detection_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = object_detection_dataset

    # test hub
    name = "test_det"
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.OBJECT_DETECTION,
        model_type="yolov8",
        model_size="n",
        categories=dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_ultralytics_classification(classification_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = classification_dataset

    # test hub
    name = "test_cls"
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.CLASSIFICATION,
        model_type="yolov8",
        model_size="n",
        categories=classification_dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_transformers_object_detection(object_detection_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = object_detection_dataset

    # test hub
    name = "test_det"
    hub = Hub.new(
        name=name,
        backend="transformers",
        task=TaskType.OBJECT_DETECTION,
        model_type="YOLOS",
        model_size="tiny",
        categories=object_detection_dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_transformers_classification(classification_dataset: Dataset, tmpdir: Path):
    image_size = 224
    dataset = classification_dataset

    # test hub
    name = "test_cls"
    hub = Hub.new(
        name=name,
        backend="transformers",
        task=TaskType.CLASSIFICATION,
        model_type="ViT",
        model_size="tiny",
        categories=classification_dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_non_hold(classification_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = classification_dataset

    # test hub
    name = "test_cls"
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.CLASSIFICATION,
        model_type="yolov8",
        model_size="n",
        categories=classification_dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size, hold=False)


def test_autocare_dlt_object_detection(object_detection_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = object_detection_dataset

    # test hub
    name = "test_det"
    hub = Hub.new(
        name=name,
        backend="autocare_dlt",
        task=TaskType.OBJECT_DETECTION,
        model_type="YOLOv5",
        model_size="s",
        categories=object_detection_dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_autocare_dlt_classification(classification_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = classification_dataset

    # temporal solution
    super_cat = [[c.supercategory, c.name] for c in dataset.get_categories()]
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
    hub = Hub.new(
        name=name,
        backend="autocare_dlt",
        task=TaskType.CLASSIFICATION,
        model_type="Classifier",
        model_size="s",
        categories=super_cat_dict_list,
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)


def test_autocare_dlt_text_recognition(text_recognition_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = text_recognition_dataset

    # test hub
    name = "test_ocr"
    hub = Hub.new(
        name=name,
        backend="autocare_dlt",
        task=TaskType.TEXT_RECOGNITION,
        model_type="TextRecognition",
        model_size="s",
        categories=dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size)
