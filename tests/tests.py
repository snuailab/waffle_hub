import time
import uuid
from pathlib import Path

import pytest

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

    name = f"test_{uuid.uuid1()}"
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.CLASSIFICATION,
        model_type="yolov8",
        model_size="n",
        categories=dataset.get_category_names(),
        device="cpu",
        workers=0,
        hold=hold,
    )

    result = hub.hpo(
        dataset,
        n_trials,
        direction,
        hpo_method,
        image_size=24,
        search_space={
            "lr0": [0.005, 0.05],
            "lrf": [0.001, 0.005],
            "mosaic": [0.6, 1],
            "cos_lr": (True, False),
            "hsv_h": [0.01, 0.02],
        },
        epochs=3,
        batch_size=24,
    )

    if not hold:
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


if __name__ == "__main__":
    test_classification_hpo()
