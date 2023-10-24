import time
import uuid
from pathlib import Path
from typing import Union

import pytest
from waffle_utils.file import io

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub
from waffle_hub.schema.configs import HPOConfig, TrainConfig
from waffle_hub.schema.result import HPOResult


def test_classification_hpo():
    n_trials = 2
    search_space = {
        "advance_params": {
            "lr0": {"method": "suggest_float", "search_space": [0.005, 0.05], "kwargs": {}},
            "lrf": {"method": "suggest_float", "search_space": [0.1, 0.5], "kwargs": {}},
        },
        "image_size": {"method": "suggest_categorical", "search_space": [64, 28], "kwargs": {}},
    }
    direction = "maximize"
    sampler = "TPESampler"
    pruner = {"medianpruner": {"n_startup_trials": 5, "n_warmup_steps": 5}}
    metric = "mAP"
    dataset = "mnist_classification"
    name = f"test_object_detection_hpo_{uuid.uuid4()}"
    hold = False
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.CLASSIFICATION,
        model_type="yolov8",
        model_size="n",
    )
    result = hub.hpo(
        dataset=dataset,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        n_trials=n_trials,
        metric="accuracy",
        search_space=search_space,
        batch_size=24,
        device="0",
        epochs=2,
        hold=False,
    )

    if not hold:
        assert hasattr(result, "callback")
        while not result.callback.is_finished():
            time.sleep(1)
        assert result.callback.is_finished()
        assert not result.callback.is_failed()
    # hpo_train_config = TrainConfig.load(Path(hub.root_dir / hub.name / "configs" / "train.yaml"))

    # train_result = hub.train(dataset=dataset, image_size=640, device="cpu")

    # hpo_config = HPOConfig.load(Path(hub.root_dir / hub.name / "configs" / "hpo.yaml"))

    # hpo_train_config = TrainConfig.load(Path(hub.root_dir / hub.name / "configs" / "train.yaml"))
    # train_hub = Hub.load(name=name)
    # train_hub: Hub = Hub.from_model_config(
    #     name=name + "_from_model_config",
    #     model_config_file=Path("hubs") / name / Hub.MODEL_CONFIG_FILE,
    # )
    # train_result = train_hub.train(
    #     dataset=dataset,
    #     epochs=hpo_train_config.epochs,
    #     batch_size=hpo_train_config.batch_size,
    #     image_size=hpo_train_config.image_size,
    #     learning_rate=hpo_train_config.learning_rate,
    #     letter_box=hpo_train_config.letter_box,
    #     device="cpu",
    #     workers=0,
    #     advance_params=hpo_train_config.advance_params,
    # )

    # assert len(train_result.metrics) >= 1
    # assert len(train_result.eval_metrics) >= 1
    # assert Path(train_result.best_ckpt_file).exists()

    # db_name = f"{hub.name}.db"
    # last_trial = n_trials - 1
    # last_trial_directory_name = f"trial_{last_trial}"

    # assert isinstance(result, HPOResult)
    # assert result.best_params is not None
    # assert result.best_score is not None
    # assert Path(hub.root_dir / hub.name / "evaluate.json").exists()
    # assert Path(hub.root_dir / hub.name / "hpo.json").exists()
    # assert Path(hub.root_dir / hub.name / "metrics.json").exists()
    # assert Path(hub.root_dir / hub.name / "hpo" / last_trial_directory_name).exists()
    # assert Path(hub.root_dir / hub.name / db_name).exists()

    # hpo_config = HPOConfig.load(Path(hub.root_dir / hub.name / "configs" / "hpo.yaml"))
    # assert hpo_config.direction.lower() == direction.lower()
    # assert len(train_result.metrics) >= 1
    # assert len(train_result.eval_metrics) >= 1
    # assert Path(train_result.best_ckpt_file).exists()

    # if train_hub.backend == "ultralytics":
    #     if train_hub.task in [TaskType.INSTANCE_SEGMENTATION, TaskType.OBJECT_DETECTION]:
    #         assert train_hub.get_train_config().letter_box == True
    #     elif train_hub.task == TaskType.CLASSIFICATION:
    #         assert train_hub.get_train_config().letter_box == False


if __name__ == "__main__":
    test_classification_hpo()
