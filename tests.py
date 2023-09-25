import time
import uuid
from pathlib import Path

import pytest
from waffle_utils.file import io

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub


def test_classification_hpo():
    hold = True
    dataset = Dataset.load(
        name="mnist_classification", root_dir="/home/daeun/workspace/waffle_hub/datasets"
    )

    n_trials = 1
    direction = "maximize"
    hpo_method = "randomSampler"

    name = f"test"
    # hub = Hub.new(
    #     name=name,
    #     backend="ultralytics",
    #     task=TaskType.CLASSIFICATION,
    #     model_type="yolov8",
    #     model_size="n",
    #     categories=dataset.get_category_names(),
    #     device="cpu",
    #     workers=0,
    #     hold=hold,
    # )
    hub = Hub.load(name=name)
    # load, new -> hpo [option]
    result = hub.hpo(
        dataset,
        n_trials,
        direction,
        hpo_method,
        search_space={
            "lr0": [0.005, 0.05],
            "lrf": [0.001, 0.005],
            "mosaic": [0.6, 1],
            "cos_lr": (True, False),
            "hsv_h": [0.01, 0.02],
        },
        image_size=24,  # TODO :
        epochs=3,
        batch_size=24,
        advance_hpo_params={},
    )

    db_name = f"{hub.name}.db"

    last_trial = n_trials - 1
    last_trial_directory_name = f"trial_{last_trial}"
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result
    assert Path(hub.root_dir / hub.name / "evaluate.json").exists()
    assert Path(hub.root_dir / hub.name / "hpo.json").exists()
    load_result = io.load_json(Path(hub.root_dir / hub.name / "hpo.json"))
    assert load_result["methods"]["direction"].lower() == direction.lower()
    assert load_result["methods"]["sampler"].lower() == hpo_method.lower()
    assert Path(hub.root_dir / hub.name / "metrics.json").exists()
    assert Path(hub.root_dir / hub.name / "hpo" / last_trial_directory_name).exists()
    assert Path(hub.root_dir / hub.name / db_name).exists()


if __name__ == "__main__":
    test_classification_hpo()
