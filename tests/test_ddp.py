import time
from pathlib import Path

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub
from waffle_hub.schema.result import TrainResult


def _train(hub, dataset: Dataset, image_size: int, hold: bool = True):
    result: TrainResult = hub.train(
        dataset=dataset,
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
    name = "test_seg_ultralytics"
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

    _train(hub, dataset, image_size)


def test_ultralytics_object_detection(object_detection_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = object_detection_dataset

    # test hub
    name = "test_det_ultralytics"
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

    _train(hub, dataset, image_size)


def test_ultralytics_classification(classification_dataset: Dataset, tmpdir: Path):
    image_size = 32
    dataset = classification_dataset

    # test hub
    name = "test_cls_ultralytics"
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

    _train(hub, dataset, image_size)


# def test_transformers_classification(classification_dataset: Dataset, tmpdir: Path):
#     image_size = 224
#     dataset = classification_dataset

#     # test hub
#     name = "test_cls_transformers"
#     hub = Hub.new(
#         name=name,
#         backend="transformers",
#         task=TaskType.CLASSIFICATION,
#         model_type="ViT",
#         model_size="tiny",
#         categories=classification_dataset.get_category_names(),
#         root_dir=tmpdir,
#     )
#     hub = Hub.load(name=name, root_dir=tmpdir)
#     hub: Hub = Hub.from_model_config(
#         name=name + "_from_model_config",
#         model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
#         root_dir=tmpdir,
#     )

#     _train(hub, dataset, image_size)


# def test_transformers_object_detection(object_detection_dataset: Dataset, tmpdir: Path):
#     image_size = 32
#     dataset = object_detection_dataset

#     # test hub
#     name = "test_det_transformers"
#     hub = Hub.new(
#         name=name,
#         backend="transformers",
#         task=TaskType.OBJECT_DETECTION,
#         model_type="YOLOS",
#         model_size="tiny",
#         categories=dataset.get_category_names(),
#         root_dir=tmpdir,
#     )
#     hub = Hub.load(name=name, root_dir=tmpdir)
#     hub: Hub = Hub.from_model_config(
#         name=name + "_from_model_config",
#         model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
#         root_dir=tmpdir,
#     )

#     _train(hub, dataset, image_size)
