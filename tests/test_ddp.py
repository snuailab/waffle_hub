import time
from pathlib import Path

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub.adapter.ultralytics import UltralyticsHub
from waffle_hub.schema.result import TrainResult


def _train(hub, dataset: Dataset, image_size: int, hold: bool = True):
    result: TrainResult = hub.train(
        dataset_path=dataset.export(hub.backend),
        epochs=1,
        image_size=image_size,
        batch_size=4,
        pretrained_model=None,
        device="0,1",
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
        categories=dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = UltralyticsHub.load(name=name, root_dir=tmpdir)
    hub: UltralyticsHub = UltralyticsHub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / UltralyticsHub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _train(hub, dataset, image_size)


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
        categories=dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = UltralyticsHub.load(name=name, root_dir=tmpdir)
    hub: UltralyticsHub = UltralyticsHub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / UltralyticsHub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _train(hub, dataset, image_size)


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
        categories=classification_dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub = UltralyticsHub.load(name=name, root_dir=tmpdir)
    hub: UltralyticsHub = UltralyticsHub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / UltralyticsHub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )

    _train(hub, dataset, image_size)
