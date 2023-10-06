import time
from pathlib import Path
from typing import Union

import pytest

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub
from waffle_hub.schema.configs import HPOConfig, TrainConfig
from waffle_hub.schema.result import HPOResult, TrainResult


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


def assert_train_result_after_hpo(train_hub: Hub, train_result: HPOResult):
    """
    after get best params by HPO and train with best params, check train result
    """
    assert len(train_result.metrics) >= 1
    assert len(train_result.eval_metrics) >= 1
    assert Path(train_result.best_ckpt_file).exists()
    if train_hub.backend == "ultralytics":
        if train_hub.task in [TaskType.INSTANCE_SEGMENTATION, TaskType.OBJECT_DETECTION]:
            assert train_hub.get_train_config().letter_box == True
        elif train_hub.task == TaskType.CLASSIFICATION:
            assert train_hub.get_train_config().letter_box == False


def assert_hpo_result(hub: Hub, hpo_result: HPOResult, n_trials: int):
    """
    check hpo result
    """
    db_name = f"{hub.name}.db"
    last_trial = n_trials - 1
    last_trial_directory_name = f"trial_{last_trial}"

    assert isinstance(hpo_result, HPOResult)
    assert hpo_result.best_params is not None
    assert hpo_result.best_score is not None
    assert Path(hub.root_dir / hub.name / "evaluate.json").exists()
    assert Path(hub.root_dir / hub.name / "hpo.json").exists()
    assert Path(hub.root_dir / hub.name / "metrics.json").exists()
    assert Path(hub.root_dir / hub.name / "hpo" / last_trial_directory_name).exists()
    assert Path(hub.root_dir / hub.name / db_name).exists()


def assert_hpo_method(
    hpo_config: HPOConfig, sampler: Union[str, dict], pruner: Union[str, dict], direction: str
):
    """
    check hpo methods
    """
    assert hpo_config.direction.lower() == direction.lower()
    if isinstance(sampler, str):
        assert hpo_config.sampler.lower() == sampler.lower()
    elif isinstance(sampler, dict):
        assert hpo_config.sampler.lower() == iter(sampler.keys()).__next__().lower()
    if isinstance(pruner, str):
        assert hpo_config.pruner.lower() == pruner.lower()
    elif isinstance(pruner, dict):
        assert hpo_config.pruner.lower() == iter(pruner.keys()).__next__().lower()


@pytest.mark.parametrize(
    "n_trials,  search_space, direction, sampler, pruner, metric",
    [
        (
            2,
            {
                "lr0": {"method": "suggest_float", "search_space": [0.1, 0.5], "kwargs": {}},
                "lrf": {"method": "suggest_float", "search_space": [0.01, 0.1], "kwargs": {}},
                "epochs": {"method": "suggest_categorical", "search_space": [1, 2, 3], "kwargs": {}},
                "batch_size": {
                    "method": "suggest_categorical",
                    "search_space": [4, 8, 16],
                    "kwargs": {},
                },
            },
            "maximize",
            "TPESampler",
            {"medianpruner": {"n_startup_trials": 5, "n_warmup_steps": 5}},
            "mAP",
        ),
    ],
)
def test_object_detection_hpo(
    object_detection_dataset: Dataset,
    tmpdir: Path,
    n_trials: int,
    search_space: dict,
    direction: str,
    sampler: Union[str, dict],
    pruner: Union[str, dict],
    metric: str,
):
    dataset = object_detection_dataset
    name = f"test_object_detection_hpo"
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.OBJECT_DETECTION,
        model_type="yolov8",
        model_size="n",
        categories=dataset.get_category_names(),
        root_dir=tmpdir,
        device="0,1",
        workers=0,
        hold=True,
    )

    hub.hpo(
        dataset=dataset,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        n_trials=n_trials,
        metric=metric,
        search_space=search_space,
        image_size=64,
    )

    hpo_train_config = TrainConfig.load(Path(hub.root_dir / hub.name / "configs" / "train.yaml"))
    train_hub = Hub.load(name=name, root_dir=tmpdir)
    train_hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )
    train_result = train_hub.train(
        dataset=dataset,
        epochs=hpo_train_config.epochs,
        batch_size=hpo_train_config.batch_size,
        image_size=hpo_train_config.image_size,
        learning_rate=hpo_train_config.learning_rate,
        letter_box=hpo_train_config.letter_box,
        device="0,1",
        workers=0,
        advance_params=hpo_train_config.advance_params,
    )

    hpo_config = HPOConfig.load(Path(hub.root_dir / hub.name / "configs" / "hpo.yaml"))
    hpo_result = HPOResult.load(Path(hub.root_dir / hub.name / "hpo.json"))
    assert_train_result_after_hpo(train_hub, train_result)
    assert_hpo_result(hub, hpo_result, n_trials)
    assert_hpo_method(hpo_config, sampler, pruner, direction)


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
