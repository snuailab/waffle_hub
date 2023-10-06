from pathlib import Path
from typing import Union

import pytest

from waffle_hub import TaskType
from waffle_hub.core.hpo.adapter.optuna import OptunaHPO
from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub
from waffle_hub.schema.configs import HPOConfig, TrainConfig
from waffle_hub.schema.result import HPOResult


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


def assert_hpo_result(
    root_dir: Path, name: str, hpo_result: HPOResult, n_trials: int, is_hub: bool = True
):
    """
    check hpo result
    """
    db_name = f"{name}.db"
    last_trial = n_trials - 1
    last_trial_directory_name = f"trial_{last_trial}"
    assert isinstance(hpo_result, HPOResult)
    assert hpo_result.best_params is not None
    assert hpo_result.best_score is not None
    assert Path(root_dir / name / "hpo.json").exists()

    if is_hub:
        assert Path(root_dir / name / "metrics.json").exists()
        assert Path(root_dir / name / "evaluate.json").exists()
        assert Path(root_dir / name / "hpo" / last_trial_directory_name).exists()
    assert Path(root_dir / name / db_name).exists()


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
            2,
            32,
        ),
        (
            2,
            "GridSampler",
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
        (
            2,
            {
                "lr0": {"method": "suggest_float", "search_space": [0.005, 0.05], "kwargs": {}},
                "lrf": {"method": "suggest_float", "search_space": [0.1, 0.5], "kwargs": {}},
                "epochs": {"method": "suggest_int", "search_space": [1, 3], "kwargs": {}},
                "batch_size": {
                    "method": "suggest_categorical",
                    "search_space": [4, 8, 16],
                    "kwargs": {},
                },
            },
            "minimize",
            {"TPESampler": {"n_startup_trials": 20, "multivariate": False}},
            "MedianPruner",
            "accuracy",
        ),
    ],
)

# TODO : Add HPO config -> train config

# def test_hpo_config(

# )


def test_object_detection_hpo(
    n_trials,
    hpo_method,
    search_space,
    direction,
    epochs,
    batch_size,
    object_detection_dataset: Dataset,
    tmpdir: Path,
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
        device="cpu",
        workers=0,
        hold=True,
    )
    hub = Hub.load(name=name, root_dir=tmpdir)
    hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )
    result = hub.hpo(
        dataset,
        n_trials,
        direction,
        hpo_method,
        search_space,
        epochs=epochs,
        batch_size=batch_size,
    )

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


def test_classification_hpo(
    n_trials,
    hpo_method,
    search_space,
    direction,
    epochs,
    batch_size,
    classification_dataset: Dataset,
    tmpdir: Path,
):

    dataset = classification_dataset
    name = f"test_{hpo_method}_{direction}"
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.CLASSIFICATION,
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

    result = hub.hpo(
        dataset,
        n_trials,
        direction,
        hpo_method,
        search_space,
        epochs=epochs,
        batch_size=batch_size,
    )

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


def test_no_hold_classification_hpo(
    n_trials,
    hpo_method,
    search_space,
    direction,
    epochs,
    batch_size,
    classification_dataset: Dataset,
    tmpdir: Path,
):

    dataset = classification_dataset
    name = f"test_{hpo_method}_{direction}"
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.CLASSIFICATION,
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

    result = hub.hpo(
        dataset,
        n_trials,
        direction,
        hpo_method,
        search_space,
        epochs=epochs,
        batch_size=batch_size,
    )

    assert hasattr(result, "callback")
    while not result.callback.is_finished() and not result.callback.is_failed():
        time.sleep(1)
    assert result.callback.is_finished()
    assert not result.callback.is_failed()
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
