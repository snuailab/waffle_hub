import json
import tempfile
import time
from pathlib import Path

import pytest
import torch

from waffle_hub import (
    EvaluateStatusDesc,
    ExportOnnxStatusDesc,
    ExportWaffleStatusDesc,
    InferenceStatusDesc,
    TaskType,
    TrainStatusDesc,
)
from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub
from waffle_hub.schema.result import (
    EvaluateResult,
    ExportOnnxResult,
    ExportWaffleResult,
    InferenceResult,
    TrainResult,
)


def _train(hub, dataset: Dataset, image_size: int, advance_params: dict = None):
    result: TrainResult = hub.train(
        dataset=dataset,
        epochs=1,
        image_size=image_size,
        batch_size=4,
        pretrained_model=None,
        letter_box=False,
        device="cpu",
        workers=0,
        advance_params=advance_params,
    )

    training_status = hub.get_training_status()
    assert training_status.status_desc == TrainStatusDesc.SUCCESS
    assert training_status.step == training_status.total_step
    assert len(result.metrics) >= 1
    assert len(result.eval_metrics) >= 1
    assert Path(result.best_ckpt_file).exists()
    # assert Path(result.last_ckpt_file).exists()

    if hub.backend == "ultralytics":
        if hub.task in [TaskType.INSTANCE_SEGMENTATION, TaskType.OBJECT_DETECTION]:
            assert hub.get_train_config().letter_box == True
        elif hub.task == TaskType.CLASSIFICATION:
            assert hub.get_train_config().letter_box == False

    return result


def _evaluate(hub, dataset: Dataset):

    result: EvaluateResult = hub.evaluate(
        dataset=dataset,
        device="cpu",
        workers=0,
    )

    evaluating_status = hub.get_evaluating_status()
    assert evaluating_status.status_desc == EvaluateStatusDesc.SUCCESS
    assert evaluating_status.step == evaluating_status.total_step
    assert len(result.eval_metrics) >= 1

    return result


def _inference(hub, source: str):

    result: InferenceResult = hub.inference(
        source=source,
        draw=True,
        device="cpu",
        workers=0,
    )

    inferencing_status = hub.get_inferencing_status()
    assert inferencing_status.status_desc == InferenceStatusDesc.SUCCESS
    assert inferencing_status.step == inferencing_status.total_step
    assert len(result.predictions) >= 1
    assert Path(result.draw_dir).exists()

    return result


def _export_onnx(hub, half: bool = False):
    result: ExportOnnxResult = hub.export_onnx(
        half=half,
        device="cpu",
    )

    exporting_onnx_status = hub.get_exporting_onnx_status()
    assert exporting_onnx_status.status_desc == ExportOnnxStatusDesc.SUCCESS
    assert Path(result.onnx_file).exists()

    return result


def _export_waffle(hub):
    result: ExportWaffleResult = hub.export_waffle()

    exporting_waffle_status = hub.get_exporting_waffle_status()
    assert exporting_waffle_status.status_desc == ExportWaffleStatusDesc.SUCCESS
    assert Path(result.waffle_file).exists()

    return result


def _from_waffle_file(waffle_file: str, source: str, tmpdir: Path):
    name = "test_import"
    hub = Hub.from_waffle_file(name=name, waffle_file=waffle_file, root_dir=tmpdir)

    _inference(hub, source)


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


def _total(
    hub,
    dataset: Dataset,
    image_size: int,
    tmpdir: Path,
    advance_params: dict = None,
):

    _train(hub, dataset, image_size, advance_params=advance_params)
    _evaluate(hub, dataset)
    _inference(hub, dataset.raw_image_dir)
    _export_onnx(hub, half=False)  # _export_onnx(hub, half=True)  # cpu cannot be half
    result = _export_waffle(hub)
    _from_waffle_file(result.waffle_file, dataset.raw_image_dir, tmpdir)
    _feature_extraction(hub, image_size)
    _benchmark(hub, image_size)
    _util(hub)


def test_ultralytics_segmentation(
    instance_segmentation_dataset: Dataset, tmpdir: Path, test_video_path: Path
):
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

    _total(hub, dataset, image_size, tmpdir)
    _inference(hub, test_video_path)


def test_ultralytics_object_detection(
    object_detection_dataset: Dataset, tmpdir: Path, test_video_path: Path
):
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

    _total(hub, dataset, image_size, tmpdir)
    _inference(hub, test_video_path)


def test_ultralytics_object_detection_advance_params(
    object_detection_dataset: Dataset, tmpdir: Path
):
    image_size = 32
    dataset = object_detection_dataset

    # test hub
    name = "test_det_adv"
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

    hub.get_default_advance_train_params()

    import_hub_name = "test_import"
    _total(hub, dataset, image_size, tmpdir, {"box": 4, "cls": 1})
    hub.delete_artifact()
    import_hub = Hub.load(name=import_hub_name, root_dir=tmpdir)
    import_hub.delete_hub()

    with open(str(tmpdir / "adv.json"), "w") as f:
        json.dump({"box": 4, "cls": 2}, f)
    _total(hub, dataset, image_size, tmpdir, str(tmpdir / "adv.json"))
    hub.delete_artifact()
    import_hub = Hub.load(name=import_hub_name, root_dir=tmpdir)
    import_hub.delete_hub()

    with pytest.raises(ValueError):
        _total(hub, dataset, image_size, tmpdir, {"box": 4, "dummy_adv_param": 2})
        assert hub.get_training_status().status_desc == TrainStatusDesc.FAILED
        import_hub = Hub.load(name=import_hub_name, root_dir=tmpdir)
        import_hub.delete_hub()


def test_ultralytics_classification(
    classification_dataset: Dataset, tmpdir: Path, test_video_path: Path
):
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

    _total(hub, dataset, image_size, tmpdir)
    _inference(hub, test_video_path)


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

    _total(hub, dataset, image_size, tmpdir)


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

    _total(hub, dataset, image_size, tmpdir)


def test_autocare_dlt_classification(classification_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = classification_dataset

    # test hub
    name = "test_cls"
    hub = Hub.new(
        name=name,
        backend="autocare_dlt",
        task=TaskType.CLASSIFICATION,
        model_type="Classifier",
        model_size="s",
        categories=dataset.get_categories(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size, tmpdir)


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
        categories=dataset.get_categories(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size, tmpdir)


def test_autocare_dlt_semantic_segmentation(semantic_segmentation_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = semantic_segmentation_dataset

    # test hub
    name = "test_segmantic_seg"
    hub = Hub.new(
        name=name,
        backend="autocare_dlt",
        task=TaskType.SEMANTIC_SEGMENTATION,
        model_type="Segmenter",
        model_size="m",
        categories=semantic_segmentation_dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size, tmpdir)


def test_autocare_dlt_classification_without_category(classification_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = classification_dataset

    # test hub
    name = "test_cls"
    hub = Hub.new(
        name=name,
        backend="autocare_dlt",
        task=TaskType.CLASSIFICATION,
        model_type="Classifier",
        model_size="s",
        # categories=dataset.get_categories(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size, tmpdir)


def test_ultralytics_classification_without_category(classification_dataset: Dataset, tmpdir: Path):
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
        # categories=classification_dataset.get_category_names(),  # auto detect
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size, tmpdir)


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
        # categories=classification_dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _total(hub, dataset, image_size, tmpdir)
