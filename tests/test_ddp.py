import time
from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    "n_trials, hpo_method, search_space, direction, epochs, batch_size",
    [
        (
            5,
            "TPESampler",
            {
                "lr0": [0.005, 0.05],
                "lrf": [0.001, 0.005],
                "mosaic": [0.6, 1],
                "cos_lr": (True, False),
                "hsv_h": [0.01, 0.02],
            },
            "maximize",
            2,
            32,
        ),
        (
            5,
            "BOHB",
            {
                "lr0": [0.005, 0.05],
                "lrf": [0.001, 0.005],
                "mosaic": [0.6, 1],
                "cos_lr": (True, False),
                "hsv_h": [0.01, 0.02],
            },
            "minimize",
            2,
            32,
        ),
    ],
)
def test_object_detection_hpo(
    object_detection_dataset: Dataset,
    tmpdir: Path,
    n_trials,
    hpo_method,
    search_space,
    direction,
    epochs,
    batch_size,
):

    dataset = object_detection_dataset
    name = f"test_{hpo_method}_{direction}"
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.OBJECT_DETECTION,
        model_type="yolov8",
        model_size="n",
        categories=dataset.get_category_names(),
        root_dir=tmpdir,
    )
    result = hub.hpo(
        dataset,
        n_trials,
        direction,
        hpo_method,
        search_space=search_space,
        epochs=epochs,
        device="0,1",
        workers=0,
        hold=True,
        batch_size=batch_size,
    )
    train_hub = Hub.load(name=name, root_dir=tmpdir)
    train_hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )
    train_result = train_hub.train(
        dataset=dataset,
        epochs=1,
        batch_size=batch_size,
        pretrained_model=None,
        letter_box=False,
        device="0,1",
        workers=0,
        hold=True,
    )

    assert len(train_result.metrics) >= 1
    assert len(train_result.eval_metrics) >= 1
    assert Path(train_result.best_ckpt_file).exists()
    # assert Path(result.last_ckpt_file).exists()
    db_name = f"{hub.name}.db"
    last_trial = n_trials - 1
    last_trial_directory_name = f"trial_{last_trial}"

    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result
    assert Path(hub.root_dir / hub.name / "evaluate.json").exists()
    assert Path(hub.root_dir / hub.name / "hpo.json").exists()
    assert Path(hub.root_dir / hub.name / "metrics.json").exists()
    assert Path(hub.root_dir / hub.name / "hpo" / last_trial_directory_name).exists()
    assert Path(hub.root_dir / hub.name / db_name).exists()

    assert len(train_result.metrics) >= 1
    assert len(train_result.eval_metrics) >= 1
    assert Path(train_result.best_ckpt_file).exists()
    if train_hub.backend == "ultralytics":
        if train_hub.task in [TaskType.INSTANCE_SEGMENTATION, TaskType.OBJECT_DETECTION]:
            assert train_hub.get_train_config().letter_box == True
        elif train_hub.task == TaskType.CLASSIFICATION:
            assert train_hub.get_train_config().letter_box == False


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
