import time
from pathlib import Path

import pytest
from waffle_utils.file import io

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub


@pytest.mark.parametrize(
    "n_trials, hpo_method, search_space, direction, epochs, batch_size",
    [
        (
            1,
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
    ],
)
def test_object_detection_hpo(
    object_detection_dataset: Dataset,
    tmpdir: Path,
    n_trials: int,
    hpo_method: str,
    search_space: dict,
    direction: str,
    epochs: int,
    batch_size: int,
):

    dataset = object_detection_dataset
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
        objective=None,
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
        device="cpu",
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
    load_result = io.load_json(Path(hub.root_dir / hub.name / "hpo.json"))
    assert load_result[0]["methods"]["direction"].lower() == direction.lower()
    assert load_result[0]["methods"]["sampler"].lower() == hpo_method.lower()
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
def test_classification_hpo(
    classification_dataset: Dataset,
    tmpdir: Path,
    n_trials: int,
    hpo_method: str,
    search_space: dict,
    direction: str,
    epochs: int,
    batch_size: int,
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
    result = hub.hpo_new(
        dataset,
        n_trials,
        direction,
        hpo_method,
        search_space=search_space,
        epochs=epochs,
        device="cpu",
        batch_size=batch_size,
    )
    train_hub = Hub.load(name=name, root_dir=tmpdir)
    train_config = train_hub.get_train_config()
    train_hub: Hub = Hub.from_model_config(
        name=name + "_from_model_config",
        model_config_file=tmpdir / name / Hub.MODEL_CONFIG_FILE,
        root_dir=tmpdir,
    )
    train_result = train_hub.train(
        dataset=train_config.dataset,
        epochs=train_config.epochs,
        batch_size=train_config.batch_size,
        letter_box=train_config.letter_box,
        device="cpu",
        workers=0,
        image_size=train_config.image_size,
        advance_params=train_config.advance_params,
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
    load_result = io.load_json(Path(hub.root_dir / hub.name / "hpo.json"))
    assert load_result[0]["methods"]["direction"].lower() == direction.lower()
    assert load_result[0]["methods"]["sampler"].lower() == hpo_method.lower()
    assert len(train_result.metrics) >= 1
    assert len(train_result.eval_metrics) >= 1
    assert Path(train_result.best_ckpt_file).exists()
    if train_hub.backend == "ultralytics":
        if train_hub.task in [TaskType.INSTANCE_SEGMENTATION, TaskType.OBJECT_DETECTION]:
            assert train_hub.get_train_config().letter_box == True
        elif train_hub.task == TaskType.CLASSIFICATION:
            assert train_hub.get_train_config().letter_box == False
