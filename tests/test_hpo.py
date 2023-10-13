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
        device="cpu",
        workers=0,
        hold=True,
    )

    train_result = hub.train(dataset=dataset)

    hpo_config = HPOConfig.load(Path(hub.root_dir / hub.name / "configs" / "hpo.yaml"))
    hpo_result = HPOResult.load(Path(hub.root_dir / hub.name / "hpo.json"))

    assert_train_result_after_hpo(hub, train_result)
    assert_hpo_result(hub.root_dir, hub.name, hpo_result, n_trials)
    assert_hpo_method(hpo_config, sampler, pruner, direction)


@pytest.mark.parametrize(
    "n_trials, search_space, direction, sampler, pruner, metric",
    [
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
def test_classification_hpo(
    classification_dataset: Dataset,
    tmpdir: Path,
    n_trials: int,
    search_space: dict,
    direction: str,
    sampler: Union[str, dict],
    pruner: Union[str, dict],
    metric: str,
):
    dataset = classification_dataset
    name = f"test_classification_hpo"
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.CLASSIFICATION,
        model_type="yolov8",
        model_size="n",
        categories=dataset.get_category_names(),
        root_dir=tmpdir,
    )
    hub.hpo(
        dataset=dataset,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        n_trials=n_trials,
        metric=metric,
        device="cpu",
        search_space=search_space,
        image_size=64,
        workers=0,
        hold=True,
    )

    train_result = hub.train(dataset=dataset)

    hpo_config = HPOConfig.load(Path(hub.root_dir / hub.name / "configs" / "hpo.yaml"))
    hpo_result = HPOResult.load(Path(hub.root_dir / hub.name / "hpo.json"))

    assert_train_result_after_hpo(hub, train_result)
    assert_hpo_result(hub.root_dir, hub.name, hpo_result, n_trials)
    assert_hpo_method(hpo_config, sampler, pruner, direction)


def simple_func(x, y, z, a, q, w, e, r, t, u, i, o):
    return x**2 + (y + z + a + q + w + e) / 2 + (r + t + y + u + i + o) * 2


@pytest.mark.parametrize(
    "n_trials,  search_space, direction, sampler, pruner",
    [
        (
            2,
            {
                "x": {
                    "method": "suggest_categorical",
                    "search_space": [1, 3, 4, 6],
                    "kwargs": {},
                },
                "y": {
                    "method": "suggest_categorical",
                    "search_space": [11, 13, 5, 16],
                    "kwargs": {},
                },
                "z": {
                    "method": "suggest_categorical",
                    "search_space": [1, 8, 21],
                    "kwargs": {},
                },
                "a": {
                    "method": "suggest_categorical",
                    "search_space": [2, 8, 4, 91],
                    "kwargs": {},
                },
                "q": {
                    "method": "suggest_int",
                    "search_space": [1, 100],
                    "kwargs": {},
                },
                "w": {
                    "method": "suggest_float",
                    "search_space": [0.005, 0.05],
                    "kwargs": {},
                },
                "e": {
                    "method": "suggest_float",
                    "search_space": [0.005, 0.05],
                    "kwargs": {},
                },
                "e": {
                    "method": "suggest_float",
                    "search_space": [0.005, 0.05],
                    "kwargs": {},
                },
                "r": {
                    "method": "suggest_float",
                    "search_space": [0.005, 0.05],
                    "kwargs": {},
                },
                "t": {
                    "method": "suggest_float",
                    "search_space": [0.005, 0.05],
                    "kwargs": {},
                },
                "u": {
                    "method": "suggest_float",
                    "search_space": [0.005, 0.05],
                    "kwargs": {},
                },
                "i": {
                    "method": "suggest_float",
                    "search_space": [0.005, 0.05],
                    "kwargs": {},
                },
                "o": {
                    "method": "suggest_float",
                    "search_space": [0.005, 0.05],
                    "kwargs": {},
                },
            },
            "maximize",
            {"TPESampler": {"n_startup_trials": 20, "multivariate": False}},
            {"MedianPruner": {}},
        ),
    ],
)
def test_simple_function_hpo(
    tmpdir: Path,
    n_trials: int,
    search_space: dict,
    direction: str,
    sampler: Union[str, dict],
    pruner: Union[str, dict],
):
    name = "simple_func_hpo"
    hpo = OptunaHPO(
        study_name=name,
        root_dir=tmpdir,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        n_trials=n_trials,
        search_space=search_space,
    )
    hpo.run_hpo(objective=simple_func)

    hpo_config = HPOConfig.load(Path(tmpdir / name / "configs" / "hpo.yaml"))
    hpo_result = HPOResult.load(Path(tmpdir / name / "hpo.json"))
    assert_hpo_result(tmpdir, name, hpo_result, n_trials, is_hub=False)
    assert_hpo_method(hpo_config, sampler, pruner, direction)
