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
        batch_size=4,
        pretrained_model=None,
        letter_box=False,
        workers=0,
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
            "maximize",
            2,
            32,
        ),
    ],
)
def test_classification_hpo(
    classification_dataset: Dataset,
    tmpdir: Path,
    n_trials,
    hpo_method,
    search_space,
    direction,
    epochs,
    batch_size,
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
    result = hub.hpo(
        dataset,
        n_trials,
        direction,
        hpo_method,
        search_space=search_space,
        epochs=epochs,
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
        batch_size=4,
        pretrained_model=None,
        letter_box=False,
        workers=0,
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
