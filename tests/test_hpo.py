import time
from pathlib import Path

import pytest

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub


@pytest.mark.parametrize(
    "n_trials, hpo_method, search_space, direction, epochs, batch_size",
    [
        (
            2,
            "RandomSampler",
            {
                "lr0": [0.005, 0.05],
                "lrf": [0.001, 0.005],
                "mosaic": [0.6, 1],
                "cos_lr": (True, False),
                "hsv_h": [0.01, 0.02],
                "hsv_s": [0.01, 0.02],
                "hsv_v": [0.01, 0.02],
                "translate": [0.09, 0.11],
                "scale": [0.45, 0.55],
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
            "maximize",
            2,
            32,
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
